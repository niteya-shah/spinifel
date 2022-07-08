import matplotlib.pyplot as plt
from mpi4py              import MPI
from matplotlib          import cm
from matplotlib.colors   import LogNorm, SymLogNorm

from scipy.ndimage import gaussian_filter


from spinifel import settings, utils, image, autocorrelation, contexts
import skopi as skp
import PyNVTX as nvtx

import numpy as np
from spinifel.sequential.autocorrelation import Merge


from spinifel import SpinifelSettings
settings = SpinifelSettings()
if settings.use_cupy:
    import os
    os.environ['CUPY_ACCELERATORS'] = "cub"

    from pycuda import gpuarray
    import pycuda.autoinit

    from cupyx.scipy.sparse.linalg import LinearOperator, cg
    from cupy.linalg import norm
    import cupy as xp
else:
    from scipy.linalg        import norm
    from scipy.sparse.linalg import LinearOperator, cg
    xp = np

class MergeMPI(Merge):

    def __init__(
            self,
            settings,
            slices_,
            pixel_position_reciprocal,
            pixel_distance_reciprocal,
            nufft):
        super().__init__(
            settings,
            slices_,
            pixel_position_reciprocal,
            pixel_distance_reciprocal,
            nufft)
        # more variables are added and changes are made to some existing ones
        # required to work with MPI
        self.comm = MPI.COMM_WORLD
        self.use_psana = settings.use_psana
        self.out_dir = settings.out_dir

        self.slices_ = slices_
        self.Mquat = settings.Mquat
        self.M_ups = self.M * 2
        self.alambda = 1
        self.Mtot = self.M ** 3
        self.rlambda = self.Mtot / self.N * \
            2 ** (self.comm.rank - self.comm.size / 2)
        self.flambda = 1e5 * pow(10, self.comm.rank - self.comm.size // 2)
        self.ref_rank = -1
        self.mult = np.pi / (self.reciprocal_extent)

        lu = np.linspace(-np.pi, np.pi, self.M)
        Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')

        Qu_ = np.around(np.sqrt(Hu_**2 + Ku_**2 + Lu_**2), 4)
        F_antisupport = Qu_ > np.pi

        # Generate an antisupport in Fourier space, which has zeros in the central
        # sphere and ones in the high-resolution corners.

        assert np.all(F_antisupport == F_antisupport[::-1, :, :])
        assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
        assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
        assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])
        self.F_antisupport = xp.array(F_antisupport)

    @nvtx.annotate("mpi/autocorrelation.py::modified", is_prefix=True)
    def setup_linops(self, H, K, L, ac_support, x0):
        H_ = H.reshape(-1) * self.mult
        K_ = K.reshape(-1) * self.mult
        L_ = L.reshape(-1) * self.mult

        ugrid_conv = self.nufft.adjoint(
            self.nuvect,
            H_,
            K_,
            L_,
            1,
            self.use_reciprocal_symmetry,
            self.M_ups)

        F_ugrid_conv_ = xp.fft.fftn(
            xp.fft.ifftshift(ugrid_conv))

        def W_matvec(uvect):
            """Define W part of the W @ x = d problem."""
            uvect_ADA = self.core_problem_convolution(
                uvect, F_ugrid_conv_, ac_support)
            uvect_FDF = self.fourier_reg(uvect, ac_support)
            uvect = self.alambda * uvect_ADA + self.rlambda * uvect + self.flambda * uvect_FDF

            return uvect

        # double precision is used for convergence with Conjugated Gradient
        W = LinearOperator(
            dtype=np.complex128,
            shape=(self.Mtot, self.Mtot),
            matvec=W_matvec)

        uvect_ADb = self.nufft.adjoint(self.nuvect_Db,
                                        H_,
                                        K_,
                                        L_,
                                        ac_support,
                                        self.use_reciprocal_symmetry,
                                        self.M).reshape(-1)
        if xp.sum(xp.isnan(uvect_ADb)) > 0:
            print(
                "Warning: nans in the adjoint calculation; intensities may be too large",
                flush=True)
        d = self.alambda * uvect_ADb + self.rlambda * x0

        return W, d

    @nvtx.annotate("mpi/autocorrelation.py::modified", is_prefix=True)
    def solve_ac(self, generation, orientations=None, ac_estimate=None):
        # ac_estimate is modified in place and hence its value changes for each
        # run
        if orientations is None:
            orientations = skp.get_random_quat(self.N_images)

        H, K, L = self.get_non_uniform_positions(orientations)
        if ac_estimate is None:
            ac_support = np.ones((self.M,) * 3)
            ac_estimate = np.zeros((self.M,) * 3)
        else:
            ac_smoothed = gaussian_filter(ac_estimate, 0.5)
            ac_support = (ac_smoothed > 1e-12)
            ac_estimate *= ac_support

        # This segment completely stalls the entire code
        # if self.comm.rank == (2 if self.use_psana else 0):
        #     idx = np.abs(L) < self.reciprocal_extent * .01
        #     plt.scatter(H[idx], K[idx], c=self.slices_[idx], s=1, norm=LogNorm())
        #     plt.axis('equal')
        #     plt.colorbar()
        #     plt.savefig(self.out_dir / f"star_{generation}.png")
        #     plt.cla()
        #     plt.clf()


        ac_estimate = xp.array(ac_estimate)
        ac_support = xp.array(ac_support)
        x0 = ac_estimate.reshape(-1)

        W, d = self.setup_linops(H, K, L, ac_support, x0)
        ret, info = cg(W, d, x0=x0, maxiter=self.maxiter,
                       callback=self.callback)

        if info != 0:
            print(f'WARNING: CG did not converge at rlambda = {self.rlambda}')

        v1 = norm(ret).get()
        v2 = norm(W * ret - d).get()

        # Rank0 gathers rlambda, solution norm, residual norm from all ranks
        summary = self.comm.gather(
            (self.comm.rank, self.rlambda, v1, v2), root=0)
        print('summary =', summary)
        if self.comm.rank == 0:
            ranks, lambdas, v1s, v2s = [np.array(el) for el in zip(*summary)]

            if generation == 0:
                idx = v1s >= np.mean(v1s)
                imax = np.argmax(lambdas[idx])
                iref = np.arange(len(ranks), dtype=int)[idx][imax]
            else:
                iref = np.argmin(v1s + v2s)
            self.ref_rank = ranks[iref]
            print(
                f"Keeping result from rank {self.ref_rank}: v1={v1s[iref]} and v2={v2s[iref]}",
                flush=True)
        else:
            self.ref_rank = -1
        self.ref_rank = self.comm.bcast(self.ref_rank, root=0)

        ac = ret.reshape((self.M,) * 3).get()
        if self.use_reciprocal_symmetry:
            assert np.all(np.isreal(ac))
        ac = np.ascontiguousarray(ac.real)
        image.show_volume(
            ac,
            self.Mquat,
            f"autocorrelation_{generation}_{self.comm.rank}.png")
        print(
            f"Rank {self.comm.rank} got AC in {self.callback.counter} iterations.",
            flush=True)
        self.comm.Bcast(ac, root=self.ref_rank)

        return ac
