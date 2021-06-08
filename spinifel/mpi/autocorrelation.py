from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np
import PyNVTX as nvtx

from matplotlib.colors import LogNorm
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

import skopi as skp

from spinifel import parms, utils, image, autocorrelation


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
def setup_linops(comm, H, K, L, data,
                 ac_support, weights, x0,
                 M, Mtot, N, reciprocal_extent,
                 alambda, rlambda, flambda,
                 use_reciprocal_symmetry):
    """Define W and d parts of the W @ x = d problem.

    W = al*A_adj*Da*A + rl*I  + fl*F_adj*Df*F
    d = al*A_adj*Da*b + rl*x0 + 0

    Where:
        A represents the NUFFT operator
        A_adj its adjoint
        I the identity
        F the FFT operator
        F_adj its atjoint
        Da, Df weights
        b the data
        x0 the initial guess (ac_estimate)
    """
    H_ = H.flatten() / reciprocal_extent * np.pi / parms.oversampling
    K_ = K.flatten() / reciprocal_extent * np.pi / parms.oversampling
    L_ = L.flatten() / reciprocal_extent * np.pi / parms.oversampling
    data_ = data.flatten()

    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.sqrt(Hu_**2 + Ku_**2 + Lu_**2)
    F_antisupport = Qu_ > np.pi / parms.oversampling
    assert np.all(F_antisupport == F_antisupport[::-1, :, :])
    assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
    assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
    assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])

    # Using upsampled convolution technique instead of ADA
    M_ups = parms.M_ups
    ugrid_conv = autocorrelation.adjoint(
        np.ones_like(data_), H_, K_, L_, 1, M_ups,
        reciprocal_extent, use_reciprocal_symmetry)
    ugrid_conv = reduce_bcast(comm, ugrid_conv)
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) / Mtot

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = autocorrelation.core_problem_convolution(
            uvect, M, F_ugrid_conv_, M_ups, ac_support, use_reciprocal_symmetry)
        if False:  # Debug/test -> make sure all cg are in sync (same lambdas)
            uvect_ADA_old = core_problem(
                 comm, uvect, H_, K_, L_, ac_support, weights, M, N,
                 reciprocal_extent, use_reciprocal_symmetry)
            assert np.allclose(uvect_ADA, uvect_ADA_old)
        uvect_FDF = autocorrelation.fourier_reg(
            uvect, ac_support, F_antisupport, M, use_reciprocal_symmetry)
        uvect = alambda*uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    W = LinearOperator(
        dtype=np.complex64,
        shape=(Mtot, Mtot),
        matvec=W_matvec)

    nuvect_Db = (data_ * weights).astype(np.float32)
    uvect_ADb = autocorrelation.adjoint(
        nuvect_Db, H_, K_, L_, ac_support, M,
        reciprocal_extent, use_reciprocal_symmetry
    ).flatten()
    uvect_ADb = reduce_bcast(comm, uvect_ADb)
    d = alambda*uvect_ADb + rlambda*x0

    return W, d


@nvtx.annotate("mpi/autocorrelation.py", is_prefix=True)
def solve_ac(generation,
             pixel_position_reciprocal,
             pixel_distance_reciprocal,
             slices_,
             orientations=None,
             ac_estimate=None):
    comm = MPI.COMM_WORLD

    M = parms.M
    Mtot = M**3
    N_images = slices_.shape[0]
    N = utils.prod(slices_.shape) # no. of pixels in slices_ stack
    Ntot = N * comm.size
    reciprocal_extent = pixel_distance_reciprocal.max()
    use_reciprocal_symmetry = True
    ref_rank = -1 

    # Generate random orientations in SO(3)
    if orientations is None:
        orientations = skp.get_random_quat(N_images)
    # Calculate hkl based on orientations
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations, pixel_position_reciprocal)

    # Set up ac
    if ac_estimate is None:
        ac_support = np.ones((M,)*3)
        ac_estimate = np.zeros((M,)*3)
    else:
        ac_smoothed = gaussian_filter(ac_estimate, 0.5)
        ac_support = (ac_smoothed > 1e-12).astype(np.float)
        ac_estimate *= ac_support
    weights = np.ones(N)

    alambda = 1
    #rlambda = Mtot/Ntot * 1e2**(comm.rank - comm.size/2)
    rlambda = Mtot/Ntot * 2**(comm.rank - comm.size/2) # MONA: use base 2 instead of 100 to avoid overflown
    flambda = 0  # 1e5 * pow(10, comm.rank - comm.size//2)
    maxiter = parms.solve_ac_maxiter

    # Log central slice L~=0
    if comm.rank == (2 if parms.use_psana else 0):
        idx = np.abs(L) < reciprocal_extent * .01
        plt.scatter(H[idx], K[idx], c=slices_[idx], s=1, norm=LogNorm())
        plt.axis('equal')
        plt.colorbar()
        plt.savefig(parms.out_dir / f"star_{generation}.png")
        plt.cla()
        plt.clf()

    def callback(xk):
        callback.counter += 1
    callback.counter = 0 # counts no. of iterations of conjugate gradient

    x0 = ac_estimate.flatten()
    W, d = setup_linops(comm, H, K, L, slices_,
                        ac_support, weights, x0,
                        M, Mtot, N, reciprocal_extent,
                        alambda, rlambda, flambda,
                        use_reciprocal_symmetry)
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)

    v1 = norm(ret)
    v2 = norm(W*ret-d)

    # Rank0 gathers rlambda, v1, v2 from all ranks
    summary = comm.gather((comm.rank, rlambda, v1, v2), root=0)
    if comm.rank == 0:
        ranks, lambdas, v1s, v2s = [np.array(el) for el in zip(*summary)]
        print("ranks =", ranks)
        print("lambdas =", lambdas)
        print("v1s =", v1s)
        print("v2s =", v2s)
        np.savez(parms.out_dir / f"summary-{generation}", ranks=ranks, lambdas=lambdas, v1s=v1s, v2s=v2s)

        if generation == 0:
            # Expect non-convergence => weird results.
            # Heuristic: retain rank with highest lambda and high v1.
            idx = v1s >= np.mean(v1s)
            imax = np.argmax(lambdas[idx])
            iref = np.arange(len(ranks), dtype=np.int)[idx][imax]
        else:
            # Take corner of L-curve: min (v1+v2)
            iref = np.argmin(v1s+v2s)
        ref_rank = ranks[iref]

        # Log L-curve plot
        fig, axes = plt.subplots(figsize=(6.0, 8.0), nrows=3, ncols=1)
        axes[0].loglog(lambdas, v1s)
        axes[0].loglog(lambdas[iref], v1s[iref], "rD")
        axes[0].set_xlabel("$\lambda_{r}$")
        axes[0].set_ylabel("$||x_{\lambda_{r}}||_{2}$")
        axes[1].loglog(lambdas, v2s)
        axes[1].loglog(lambdas[iref], v2s[iref], "rD")
        axes[1].set_xlabel("$\lambda_{r}$")
        axes[1].set_ylabel("$||W \lambda_{r}-d||_{2}$")
        axes[2].loglog(v2s, v1s) # L-curve
        axes[2].loglog(v2s[iref], v1s[iref], "rD")
        axes[2].set_xlabel("Residual norm $||W \lambda_{r}-d||_{2}$")
        axes[2].set_ylabel("Solution norm $||x_{\lambda_{r}}||_{2}$")
        fig.tight_layout()
        plt.savefig(parms.out_dir / f"summary_{generation}.png")
        plt.close('all')

    # Rank0 broadcasts rank with best autocorrelation
    ref_rank = comm.bcast(ref_rank, root=0)

    # Set up ac volume
    ac = ret.reshape((M,)*3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac))
    ac = np.ascontiguousarray(ac.real).astype(np.float32)
    print(f"Rank {comm.rank} got AC in {callback.counter} iterations.", flush=True)

    # Log autocorrelation volume
    image.show_volume(ac, parms.Mquat, f"autocorrelation_{generation}_{comm.rank}.png")

    # Rank[ref_rank] broadcasts its autocorrelation
    comm.Bcast(ac, root=ref_rank)
    if comm.rank == 0:
        print(f"Keeping result from rank {ref_rank}.")

    return ac
