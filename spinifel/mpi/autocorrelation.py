import matplotlib.pyplot as plt
from mpi4py              import MPI
from matplotlib          import cm
from matplotlib.colors   import LogNorm, SymLogNorm
# from scipy.linalg        import norm
from scipy.ndimage       import gaussian_filter
# from scipy.sparse.linalg import LinearOperator, cg

from spinifel import image


import os
os.environ['CUPY_ACCELERATORS'] = "cub,cutensor"

from pycuda import gpuarray
import pycuda.autoinit

import skopi as skp
import PyNVTX as nvtx

from cufinufft import cufinufft

from cupyx.scipy.sparse.linalg import LinearOperator, cg
from cupyx.scipy.linalg import norm

import cupy as cp
from scipy.ndimage import gaussian_filter
import numpy as np
from spinifel.sequential.autocorrelation import Merge

class MergeMPI(Merge):

    def __init__(
            self,
            settings,
            slices_,
            pixel_position_reciprocal,
            pixel_distance_reciprocal):
        super().__init__(settings, slices_, pixel_position_reciprocal, pixel_distance_reciprocal)
        self.comm = MPI.COMM_WORLD
        self.use_psana = settings.use_psana
        self.out_dir = settings.out_dir

        self.slices_ = slices_

        self.alambda = 1
        self.rlambda = self.Mtot/self.N * 2 **(self.comm.rank - self.comm.size/2)
        self.flambda = 1e5 * pow(10, self.comm.rank - self.comm.size//2)

    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def solve_ac(self, generation, orientations = None, ac_estimate = None):
        # ac_estimate is modified in place and hence its value changes for each run
        if orientations is None:
            orientations = skp.get_random_quat(self.N_images)

        H, K, L = self.get_non_uniform_positions(orientations)
        if ac_estimate is None:
            ac_support = np.ones((self.M,) * 3)
            ac_estimate = np.zeros((self.M,) * 3)
        else:
            ac_smoothed = gaussian_filter(ac_estimate, 0.5)
            ac_support = (ac_smoothed > 1e-12).astype(np.float)
            ac_estimate *= ac_support

        if self.comm.rank == (2 if self.use_psana else 0):
            idx = np.abs(L) < self.reciprocal_extent * .01
            plt.scatter(H[idx], K[idx], c=self.slices_[idx], s=1, norm=LogNorm())
            plt.axis('equal')
            plt.colorbar()
            plt.savefig(self.out_dir / f"star_{generation}.png")
            plt.cla()
            plt.clf()

        ref_rank = -1
        ac_estimate = cp.array(ac_estimate)
        ac_support = cp.array(ac_support)
        x0 = ac_estimate.reshape(-1)

        W, d = self.setup_linops(H, K, L, ac_support, x0)
        ret, info = cg(W, d, x0=x0, maxiter=self.maxiter,
                       callback=self.callback)

        if info != 0:
            print(f'WARNING: CG did not converge at rlambda = {self.rlambda}')

        v1 = norm(ret).get()
        v2 = norm(W*ret-d).get()

        # Rank0 gathers rlambda, solution norm, residual norm from all ranks
        summary = self.comm.gather((self.comm.rank, self.rlambda, v1, v2), root=0)
        print('summary =', summary)
        if self.comm.rank == 0:
            ranks, lambdas, v1s, v2s = [np.array(el) for el in zip(*summary)]
            
            if generation == 0:
                idx = v1s >= np.mean(v1s)
                imax = np.argmax(lambdas[idx])
                iref = np.arange(len(ranks), dtype=int)[idx][imax]
            else:
                iref = np.argmin(v1s+v2s)
            ref_rank = ranks[iref]
            print(f"Keeping result from rank {ref_rank}: v1={v1s[iref]} and v2={v2s[iref]}", flush=True)
        else:
            ref_rank = -1
        ref_rank = self.comm.bcast(ref_rank, root=0)

        ac = ret.reshape((self.M,) * 3).get()
        if self.use_reciprocal_symmetry:
            assert np.all(np.isreal(ac))
        ac = np.ascontiguousarray(ac.real)
        image.show_volume(ac, self.Mquat, f"autocorrelation_{generation}_{self.comm.rank}.png") 
        print(f"Rank {self.comm.rank} got AC in {self.callback.counter} iterations.", flush=True)
        self.comm.Bcast(ac, root=ref_rank)

        return ac
