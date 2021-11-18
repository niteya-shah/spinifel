import matplotlib.pyplot as plt
import numpy.matlib
import numpy as np
import PyNVTX as nvtx
import skopi as skp

from mpi4py              import MPI
from matplotlib.colors   import LogNorm
from scipy.linalg        import norm
from scipy.ndimage       import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

from spinifel import settings, utils, image, autocorrelation, contexts

        
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
                 M, N, reciprocal_extent,
                 rlambda, flambda,
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
    H_ = H.flatten() / reciprocal_extent * np.pi / settings.oversampling
    K_ = K.flatten() / reciprocal_extent * np.pi / settings.oversampling
    L_ = L.flatten() / reciprocal_extent * np.pi / settings.oversampling

    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.sqrt(Hu_**2 + Ku_**2 + Lu_**2)
    F_antisupport = Qu_ > np.pi / settings.oversampling
    assert np.all(F_antisupport == F_antisupport[::-1, :, :])
    assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
    assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
    assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])

    # Using upsampled convolution technique instead of ADA
    M_ups = settings.M_ups
    ugrid_conv = autocorrelation.adjoint(
        np.ones_like(data), H_, K_, L_, 1, M_ups,
        reciprocal_extent, use_reciprocal_symmetry)
    ugrid_conv = reduce_bcast(comm, ugrid_conv)
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) #/ M**3

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = autocorrelation.core_problem_convolution(
            uvect, M, F_ugrid_conv_, M_ups, ac_support, use_reciprocal_symmetry)
        if False:  # Debug/test -> make sure all cg are in sync (same lambdas)
            uvect_ADA_old = core_problem(
                 comm, uvect, H_, K_, L_, ac_support, weights, M, N,
                 reciprocal_extent, use_reciprocal_symmetry)
            print('np.linalg.norm(uvect_ADA) =', np.linalg.norm(uvect_ADA))
            print('np.linalg.norm(uvect_ADA_old) =', np.linalg.norm(uvect_ADA_old))
            print('np.linalg.norm(uvect_ADA) / np.linalg.norm(uvect_ADA_old) =', np.linalg.norm(uvect_ADA) / np.linalg.norm(uvect_ADA_old))     #assert np.allclose(uvect_ADA, uvect_ADA_old)            
        uvect_FDF = autocorrelation.fourier_reg(
            uvect, ac_support, F_antisupport, M, use_reciprocal_symmetry)
        uvect = uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    W = LinearOperator(
        dtype=np.complex128,
        shape=(M**3, M**3),
        matvec=W_matvec)

    nuvect_Db = (data * weights).astype(np.float64)
    uvect_ADb = autocorrelation.adjoint(
        nuvect_Db, H_, K_, L_, ac_support, M,
        reciprocal_extent, use_reciprocal_symmetry
    ).flatten()
    
    uvect_ADb = reduce_bcast(comm, uvect_ADb)
    
    d = uvect_ADb + rlambda*x0
  
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
    N_images = slices_.shape[0] # N images per rank
    print('N_images =', N_images)
    N = utils.prod(slices_.shape) # N images per rank x number of pixels per image = number of pixels per rank
    print('N =', N)
    Ntot = N * comm.size # total number of pixels
    print('Ntot =', Ntot)
    reciprocal_extent = pixel_distance_reciprocal.max()
    use_reciprocal_symmetry = True
    ref_rank = -1 

    # Generate random orientations in SO(3)
    if orientations is None:
        orientations = skp.get_random_quat(N_images)
    # Calculate hkl based on orientations
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations, pixel_position_reciprocal)

    # norm(nuvect_Db)^2 = b_squared = b_1^2 + b_2^2 +....
    b_squared = np.sum( np.linalg.norm( slices_.reshape(slices_.shape[0],-1) , axis=-1) **2)
    b_squared = reduce_bcast(comm, b_squared)

    data = slices_.flatten()

    # Set up ac
    if ac_estimate is None:
        ac_support = np.ones((M,)*3)
        ac_estimate = np.zeros((M,)*3)
    else:
        ac_smoothed = gaussian_filter(ac_estimate, 0.5)
        ac_support = (ac_smoothed > 1e-12).astype(np.float64)
        ac_estimate *= ac_support
    weights = np.ones(N)

    # Use scalable heuristic for regularization lambda
    rlambda = np.logspace(-10, 10, comm.size)[comm.rank]
    flambda = 0  # 1e5 * pow(10, comm.rank - comm.size//2)
    maxiter = settings.solve_ac_maxiter

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
    W, d = setup_linops(comm, H, K, L, slices_,
                        ac_support, weights, x0,
                        M, N, reciprocal_extent,
                        rlambda, flambda,
                        use_reciprocal_symmetry)
    
    # W_0 and d_0 are given by defining W and d with rlambda=0
    W_0, d_0 = setup_linops(comm, H, K, L, data,
                        ac_support, weights, x0,
                        M, N, reciprocal_extent,
                        0, flambda,
                        use_reciprocal_symmetry)

    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    print('info =', info)
    if info != 0:
        print(f'WARNING: CG did not converge at rlambda = {rlambda}')

    soln = np.linalg.norm(ret)**2 # solution norm
    resid = np.dot(ret.real,W_0.matvec(ret.real)-2*d_0) + b_squared # residual norm
    print('np.linalg.norm(W_0x) =', np.linalg.norm(W_0.matvec(ret.real)))
    print('np.linalg.norm(d_0) =', np.linalg.norm(d_0))
    print('np.linalg.norm(W_0x-d_0) =', np.linalg.norm(W_0.matvec(ret.real)-d_0)) # should be zero up to roundoff error 
    print('dot(x,W_0x-2d_0) =', np.dot(ret.real,W_0.matvec(ret.real)-2*d_0))
    print('b_squared =', b_squared)
    print('soln =', soln)
    print('resid =', resid)
    H_ = H.flatten() / reciprocal_extent * np.pi / settings.oversampling
    K_ = K.flatten() / reciprocal_extent * np.pi / settings.oversampling
    L_ = L.flatten() / reciprocal_extent * np.pi / settings.oversampling
    forward_x = autocorrelation.forward(ret.reshape((M,)*3), H_, K_, L_, 1, settings.M, H_.shape[0], reciprocal_extent, True).real
    resid_prime = np.linalg.norm(forward_x-data)**2
    print('resid_prime =', resid_prime)

    # Rank0 gathers rlambda, solution norm, residual norm from all ranks
    summary = comm.gather((comm.rank, rlambda, info, soln, resid), root=0)
    print('summary =', summary)
    if comm.rank == 0:
        ranks, lambdas, infos, solns, resids = [np.array(el) for el in zip(*summary)]
        converged_idx = [i for i, info in enumerate(infos) if info == 0]
        ranks = np.array(ranks)[converged_idx]
        lambdas = np.array(lambdas)[converged_idx]
        solns = np.array(solns)[converged_idx]
        resids = np.array([item for sublist in resids for item in sublist])[converged_idx]
        print('ranks =', ranks)
        print('lambdas =', lambdas)
        print('solns =', solns)
        print('resids =', resids)

        if generation == 0:
            # Expect non-convergence => weird results
            # Heuristic: retain rank with highest rlambda and high solution norm
            idx = solns >= np.mean(solns)
            imax = np.argmax(lambdas[idx])
            iref = np.arange(len(ranks), dtype=int)[idx][imax]
        else:
            # Heuristic: L-curve criterion
            # Take corner of L-curve
            valuePair = np.array([resids, solns]).T
            valuePair = np.array(sorted(valuePair , key=lambda k: [k[0], k[1]])) # sort the corner candidates in increasing order
            lambdas = np.sort(lambdas) # sort lambdas in increasing order
            allCoord = np.log(valuePair) # coordinates of the loglog L-curve
            nPoints = len(resids)
            if nPoints == 0: print('WARNING: no converged solution')
            firstPoint = allCoord[0]
            lineVec = allCoord[-1] - allCoord[0]
            lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
            vecFromFirst = allCoord - firstPoint
            scalarProduct = np.sum(vecFromFirst * numpy.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
            vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
            vecToLine = vecFromFirst - vecFromFirstParallel
            distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
            iref = np.argmax(distToLine)
            print('iref =', iref)
            ref_rank = ranks[iref]

            # Log L-curve plot
            fig, axes = plt.subplots(figsize=(10.0, 24.0), nrows=3, ncols=1)
            axes[0].loglog(lambdas, valuePair.T[1])
            axes[0].loglog(lambdas[iref], valuePair[iref][1], "rD")
            axes[0].set_xlabel("$\lambda$")
            axes[0].set_ylabel("Solution norm $||x||_{2}$")
            axes[1].loglog(lambdas, valuePair.T[0])
            axes[1].loglog(lambdas[iref], valuePair[iref][0], "rD")
            axes[1].set_xlabel("$\lambda$")
            axes[1].set_ylabel("Residual norm $||Ax-b||_{2}$")
            axes[2].loglog(valuePair.T[0], valuePair.T[1]) # L-curve
            axes[2].loglog(valuePair[iref][0], valuePair[iref][1], "rD")
            axes[2].set_xlabel("Residual norm $||Ax-b||_{2}$")
            axes[2].set_ylabel("Solution norm $||x||_{2}$")
            fig.tight_layout()
            plt.savefig(settings.out_dir / f"summary_{generation}.png")
            plt.close('all')

        ref_rank = ranks[iref]

    else:
        ref_rank = -1

    # Rank0 broadcasts rank with best autocorrelation
    ref_rank = comm.bcast(ref_rank, root=0)

    # Set up ac volume
    ac = ret.reshape((M,)*3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac))
    ac = np.ascontiguousarray(ac.real).astype(np.float64)
    print(f"Rank {comm.rank} got AC in {callback.counter} iterations.", flush=True)

    # Log autocorrelation volume
    image.show_volume(ac, settings.Mquat, f"autocorrelation_{generation}_{comm.rank}.png")

    # Rank[ref_rank] broadcasts its autocorrelation
    comm.Bcast(ac, root=ref_rank)
    if comm.rank == 0:
        print(f"Keeping result from rank {ref_rank}.")

    return ac
