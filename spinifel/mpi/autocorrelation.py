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
    os.environ['CUPY_ACCELERATORS'] = "cub,cutensor"

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

        """
        Generate an antisupport in Fourier space, which has zeros in the central
        sphere and ones in the high-resolution corners.

        :param M: length of the cubic antisupport volume
        :return F_antisupport: volume that masks central region
        """

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

@nvtx.annotate("mpi/autocorrelation.py", is_prefix=True)
def reduce_bcast(comm, vect):
    vect = np.ascontiguousarray(vect)
    reduced_vect = np.zeros_like(vect)
    comm.Reduce(vect, reduced_vect, op=MPI.SUM, root=0)
    vect = reduced_vect
    comm.Bcast(vect, root=0)
    return vect


@nvtx.annotate("mpi/autocorrelation.py", is_prefix=True)
def core_problem(comm, uvect, H_, K_, L_, ac_support, weights, M, N,
                 reciprocal_extent, use_reciprocal_symmetry):
    comm.Bcast(uvect, root=0)
    uvect_ADA = autocorrelation.core_problem(
        uvect, H_, K_, L_, ac_support, weights, M, N,
        reciprocal_extent, use_reciprocal_symmetry)
    uvect_ADA = reduce_bcast(comm, uvect_ADA)
    return uvect_ADA

@nvtx.annotate("mpi/autocorrelation.py", is_prefix=True)
def gen_F_antisupport(M):
    """
    Generate an antisupport in Fourier space, which has zeros in the central
    sphere and ones in the high-resolution corners.

    :param M: length of the cubic antisupport volume
    :return F_antisupport: volume that masks central region
    """
    # generate "antisupport" -- this has zeros in central sphere, 1s outside
    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.around(np.sqrt(Hu_**2 + Ku_**2 + Lu_**2), 4)
    F_antisupport = Qu_ > np.pi

    assert np.all(F_antisupport == F_antisupport[::-1, :, :])
    assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
    assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
    assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])

    return F_antisupport

@nvtx.annotate("mpi/autocorrelation.py", is_prefix=True)
def fourier_reg(uvect, support, F_antisupport, M, use_recip_sym):
    """
    Generate the flattened matrix component that penalizes noise in the outer
    regions of reciprocal space, specifically outside the unit sphere of radius
    pi, where H_max, K_max, and L_max have been normalized to equal pi.

    :param uvect: data vector on uniform grid, flattened
    :param support: 3d support object for autocorrelation
    :param F_antisupport: support in Fourier space, unmasked at high frequencies
    :param M: length of data vector along each axis
    :param use_recip_sym: if True, discard imaginary component
    :return uvect: convolution of uvect and F_antisupport, flattened
    """
    ugrid = uvect.reshape((M,) * 3) * support
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))
    F_ugrid = np.fft.fftn(np.fft.ifftshift(ugrid))
    F_reg = F_ugrid * np.fft.ifftshift(F_antisupport)
    reg = np.fft.fftshift(np.fft.ifftn(F_reg))
    uvect = (reg * support).flatten()
    if use_recip_sym:
        uvect = uvect.real
    return uvect

@nvtx.annotate("mpi/autocorrelation.py", is_prefix=True)
def setup_linops(comm, H, K, L, data,
                 ac_support, weights, x0,
                 M, Mtot, N, reciprocal_extent,
                 alambda, rlambda, flambda,
                 use_reciprocal_symmetry):
    """Define W and d parts of the W @ x = d problem.

    W = A_adj*Da*A + rl*I
    d = A_adj*Da*b + rl*x0

    Where:
        A represents the NUFFT operator
        A_adj its adjoint
        I the identity
        D weights
        b the data
        x0 the initial guess (ac_estimate)
    """
    H_ = H.flatten() / reciprocal_extent * np.pi
    K_ = K.flatten() / reciprocal_extent * np.pi
    L_ = L.flatten() / reciprocal_extent * np.pi

    F_antisupport = gen_F_antisupport(M)

    # Using upsampled convolution technique instead of ADA
    M_ups = M * 2
    ugrid_conv = autocorrelation.adjoint(
        np.ones_like(data), H_, K_, L_, M_ups,
        use_reciprocal_symmetry, support=None)
    #ugrid_conv = reduce_bcast(comm, ugrid_conv)
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv))  # / M**3

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = autocorrelation.core_problem_convolution(
            uvect, M, F_ugrid_conv_, M_ups, ac_support)
        uvect_FDF = fourier_reg(
            uvect,
            ac_support,
            F_antisupport,
            M,
            use_recip_sym=use_reciprocal_symmetry)
        uvect = alambda * uvect_ADA + rlambda * uvect + flambda * uvect_FDF
        return uvect

    W = LinearOperator(
        dtype=np.complex64,
        shape=(Mtot, Mtot),
        matvec=W_matvec)

    nuvect_Db = data * weights
    uvect_ADb = autocorrelation.adjoint(
        nuvect_Db, H_, K_, L_, M, support=ac_support,
        use_recip_sym=use_reciprocal_symmetry).flatten()

    #uvect_ADb = reduce_bcast(comm, uvect_ADb)

    if np.sum(np.isnan(uvect_ADb)) > 0:
        print(
            "Warning: nans in the adjoint calculation; intensities may be too large",
            flush=True)

    d = alambda * uvect_ADb + rlambda * x0

    return W, d

@nvtx.annotate("mpi/autocorrelation.py", is_prefix=True)
def solve_ac(generation,
             pixel_position_reciprocal,
             pixel_distance_reciprocal,
             slices_,
             orientations=None,
             ac_estimate=None):
    comm = MPI.COMM_WORLD

    M = settings.M
    Mtot = M**3
    N_images = slices_.shape[0] # N images per rank
    # N images per rank x number of pixels per image = number of pixels per
    # rank
    N = np.prod(slices_.shape)
    reciprocal_extent = pixel_distance_reciprocal.max()
    use_reciprocal_symmetry = True
    ref_rank = -1

    # Generate random orientations in SO(3)
    if orientations is None:
        orientations = skp.get_random_quat(N_images)
    # Calculate hkl based on orientations
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations, pixel_position_reciprocal)

    data = slices_.flatten().astype(np.float32)

    # Set up ac
    if ac_estimate is None:
        ac_support = np.ones((M,) * 3)
        ac_estimate = np.zeros((M,) * 3)
    else:
        ac_smoothed = gaussian_filter(ac_estimate, 0.5)
        ac_support = (ac_smoothed > 1e-12).astype(np.float)
        ac_estimate *= ac_support

    weights = np.ones(N).astype(np.float32)

    # Use scalable heuristic for regularization lambda
    alambda = 1
    rlambda = Mtot / N * 2 ** (comm.rank - comm.size / 2)
    flambda = 1e5 * pow(10, comm.rank - comm.size // 2)
    maxiter = 100

    # Log central slice L~=0
    if comm.rank == (2 if settings.use_psana else 0):
        idx = np.abs(L) < reciprocal_extent * .01
        plt.scatter(H[idx], K[idx], c=slices_[idx], s=1, norm=LogNorm())
        plt.axis('equal')
        plt.colorbar()
        plt.savefig(settings.out_dir / f"star_{generation}.png")
        plt.cla()
        plt.clf()

    def callback(xk):
        callback.counter += 1
    callback.counter = 0 # counts no. of iterations of conjugate gradient

    x0 = ac_estimate.flatten()

    W, d = setup_linops(comm, H, K, L, data,
                        ac_support, weights, x0,
                        M, Mtot, N, reciprocal_extent,
                        alambda, rlambda, flambda,
                        use_reciprocal_symmetry)

    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)

    if info != 0:
        print(f'WARNING: CG did not converge at rlambda = {rlambda}')

    v1 = norm(ret)
    v2 = norm(W * ret - d)

    # Rank0 gathers rlambda, solution norm, residual norm from all ranks
    summary = comm.gather((comm.rank, rlambda, v1, v2), root=0)
    print('summary =', summary)
    if comm.rank == 0:
        ranks, lambdas, v1s, v2s = [np.array(el) for el in zip(*summary)]

        if generation == 0:
            idx = v1s >= np.mean(v1s)
            imax = np.argmax(lambdas[idx])
            iref = np.arange(len(ranks), dtype=int)[idx][imax]
        else:
            iref = np.argmin(v1s + v2s)
        ref_rank = ranks[iref]
        print(
            f"Keeping result from rank {ref_rank}: v1={v1s[iref]} and v2={v2s[iref]}",
            flush=True)
    else:
        ref_rank = -1
    ref_rank = comm.bcast(ref_rank, root=0)

    ac = ret.reshape((M,) * 3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac))
    ac = np.ascontiguousarray(ac.real)

    # Log autocorrelation volume
    image.show_volume(
        ac,
        settings.Mquat,
        f"autocorrelation_{generation}_{comm.rank}.png")

    print(
        f"Rank {comm.rank} got AC in {callback.counter} iterations.",
        flush=True)
    comm.Bcast(ac, root=ref_rank)
    return ac
