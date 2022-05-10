import matplotlib.pyplot as plt
import numpy.matlib
import numpy as np
import PyNVTX as nvtx
import skopi as skp

from mpi4py              import MPI
from matplotlib          import cm
from matplotlib.colors   import LogNorm, SymLogNorm
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
def setup_linops(comm, generation, H, K, L, data,
                 ac_support, weights, x0,
                 M, N, reciprocal_extent,
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
    print('BEFORE: H_.shape =', H_.shape)
    print('BEFORE: K_.shape =', K_.shape)
    print('BEFORE: L_.shape =', L_.shape)
    print('np.min(H_) =', np.min(H_))
    print('np.max(H_) =', np.max(H_))
    q_ = np.sqrt(H_**2+K_**2+L_**2)
    print('q_ =', q_)

    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.sqrt(Hu_**2+Ku_**2+Lu_**2)
    print('Qu_.shape', Qu_.shape)

    F_antisupport = Qu_ > np.pi / settings.oversampling
    assert np.all(F_antisupport == F_antisupport[::-1, :, :])
    assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
    assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
    assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])

    ### weight the old data to have uniform weighted density, i.e., set the first N points to something like 1/(N_images/(2*|q|)) and set the rest of the weights to 1
    #weights /= (settings.N_images_per_rank/(2*np.abs(q_)))
    weights *= (2*np.abs(q_)/settings.N_images_per_rank)
    M_ups = settings.M_ups
    ugrid_conv = autocorrelation.adjoint(
            np.ones_like(data)*weights, H_, K_, L_, 1, M_ups,
            reciprocal_extent, use_reciprocal_symmetry)
    ugrid_conv = reduce_bcast(comm, ugrid_conv)
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) #/ M**3
    F_ugrid_conv = np.fft.fftshift(F_ugrid_conv_)

    ### 1-1) expand q regions to the H, K, L arrays
    ### take care of the corners;
    ### if there's a beamstep, add an extra < inequality for the missing center
    #q_fake = np.where(Qu_[:,:,:]>np.pi/settings.oversampling)
    q_fake = np.where(np.abs(F_ugrid_conv)<=0.1*np.max(np.abs(F_ugrid_conv)))
    #q_fake = np.where(np.abs(F_ugrid_conv)<=0.5*np.max(np.abs(F_ugrid_conv)))
    print('q_fake =', q_fake)
    print(f'# of fake q-points =', len(q_fake[0]))
    Hf_ = 2*np.pi*q_fake[0]/(M_ups-1)-np.pi
    Kf_ = 2*np.pi*q_fake[1]/(M_ups-1)-np.pi
    Lf_ = 2*np.pi*q_fake[2]/(M_ups-1)-np.pi
    print('Hf_.shape =', Hf_.shape)
    print('Kf_.shape =', Kf_.shape)
    print('Lf_.shape =', Lf_.shape)
    print('np.min(Hf_) =', np.min(Hf_))
    print('np.max(Hf_) =', np.max(Hf_))

    ### original copy of H_, K_, L_
    H_temp, K_temp, L_temp = H_, K_, L_

    ### 1-2) append Hf, Kf, Lf to H, K, L
    H_ = np.append(H_, Hf_)
    K_ = np.append(K_, Kf_)
    L_ = np.append(L_, Lf_)
    print('AFTER: H_.shape =', H_.shape)
    print('AFTER: K_.shape =', K_.shape)
    print('AFTER: L_.shape =', L_.shape)

    ugrid = x0.reshape((M,)*3) # initial guess of the autocorrelation or the autocorrelation from the previous generation
    print('np.linalg.norm(ugrid) =', np.linalg.norm(ugrid))

    ### 2) fill in the expanded regions with data sliced from old autocorrelation
    dataf = autocorrelation.forward(ugrid, Hf_, Kf_, Lf_, 1, M, Hf_.shape[0], reciprocal_extent, use_reciprocal_symmetry)
    dataf = np.zeros_like(dataf) # added
    print('dataf.shape =', dataf.shape)
    
    ### original copy of data
    data_temp = data
    print('BEFORE: data.shape =', data.shape)

    ### append dataf to data
    data = np.append(data, dataf)
    print('AFTER: data.shape =', data.shape)

    ### 3) apply different weights to different regions
    qf_ = np.sqrt(Hf_**2+Kf_**2+Lf_**2)
    print('qf_.shape =', qf_.shape)
    print('weights =', weights)
    print('np.max(weights) =', np.max(weights))
    weightsf = np.zeros(Hf_.shape[0]) # <- turn off weightsf
    weightsf = np.ones(Hf_.shape[0]) # <- turn on weightsf
    print('weightsf =', weightsf)
    num_qpoints = (settings.M_ups)**3 - Hf_.shape[0]
    print('num_qpoints = ', num_qpoints)
    num_qpoints_f = Hf_.shape[0]
    print('num_qpoints_f =', num_qpoints_f)
    scaling_factor = num_qpoints_f/num_qpoints * np.linalg.norm(weights)/np.linalg.norm(weightsf)
    scaling_factor /= comm.size
    print('scaling_factor =', scaling_factor)
    weights /= scaling_factor
    print('weights =', weights)

    ones_temp = np.ones_like(data_temp)
    onesf = np.ones_like(dataf)
    print('ones_temp.shape =', ones_temp.shape)
    print('onesf.shape =', onesf.shape)
    nuvect_ones = np.append(ones_temp * weights, onesf * weightsf)
    print('nuvect_ones.shape =', nuvect_ones.shape)

    ### compute type-1 NUFFT (nonuniform to uniform grid) with the nonuniform data replaced with 1's
    ugrid_conv = autocorrelation.adjoint(
            nuvect_ones, H_, K_, L_, 1, M_ups, 
            reciprocal_extent, use_reciprocal_symmetry)
    ugrid_conv = reduce_bcast(comm, ugrid_conv)
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) #/ M**3
    F_ugrid_conv = np.fft.fftshift(F_ugrid_conv_)

    if comm.rank == 0:
        image.show_volume(F_ugrid_conv.real, settings.Mquat*2, f"F_ugrid_conv_{generation}.png")

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = autocorrelation.core_problem_convolution(
            uvect, M, F_ugrid_conv_, M_ups, ac_support, use_reciprocal_symmetry)
        if False:  # Debug/test -> make sure all cg are in sync (same lambdas)
            print('# Debug/test')
            uvect_ADA_old = core_problem(
                 comm, uvect, H_, K_, L_, ac_support, weights, M, N,
                 reciprocal_extent, use_reciprocal_symmetry)
            print('np.linalg.norm(uvect_ADA) =', np.linalg.norm(uvect_ADA))
            print('np.linalg.norm(uvect_ADA_old) =', np.linalg.norm(uvect_ADA_old))
            print('np.linalg.norm(uvect_ADA) / np.linalg.norm(uvect_ADA_old) =', np.linalg.norm(uvect_ADA) / np.linalg.norm(uvect_ADA_old))
            #assert np.allclose(uvect_ADA, uvect_ADA_old)            
        uvect = uvect_ADA
        return uvect

    W = LinearOperator(
        dtype=np.complex128,
        shape=(M**3, M**3),
        matvec=W_matvec)

    nuvect_Db = np.append(data_temp * weights, dataf * weightsf).astype(np.float64)
    print('nuvect_Db.shape =', nuvect_Db.shape)

    uvect_ADb = autocorrelation.adjoint(
        nuvect_Db, H_, K_, L_, ac_support, M,
        reciprocal_extent, use_reciprocal_symmetry
    ).flatten()
    
    uvect_ADb = reduce_bcast(comm, uvect_ADb)
    
    d = uvect_ADb

    ### 4) reduce H, K, L and data arrays back to their original sizes
    H_, K_, L_ = H_temp, K_temp, L_temp
    data = data_temp
  
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
    N = int(utils.prod(slices_.shape)) # N images per rank x number of pixels per image = number of pixels per rank
    print('N =', N)
    reciprocal_extent = pixel_distance_reciprocal.max()
    use_reciprocal_symmetry = True

    # Generate random orientations in SO(3)
    if orientations is None:
        orientations = skp.get_random_quat(N_images)

    # Calculate hkl based on orientations
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations, pixel_position_reciprocal)

    data = slices_.flatten()
    
    # Set up ac
    if ac_estimate is None:
        ac_support = np.ones((M,)*3)
        ac_support = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ac_support)).real)).real
        ac_estimate = np.zeros((M,)*3)
        ac_estimate = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ac_estimate)).real)).real
    else:
        ac_smoothed = gaussian_filter(ac_estimate, 0.5)
        ac_support = (ac_smoothed > 1e-12).astype(np.float64)
        ac_support = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ac_support)).real)).real
        ac_estimate = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ac_estimate)).real)).real
        ac_estimate *= ac_support

    weights = np.ones(N)
    
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

    W, d = setup_linops(comm, generation, H, K, L, data,
                        ac_support, weights, x0,
                        M, N, reciprocal_extent,
                        use_reciprocal_symmetry)
    
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    print('ret =', ret)
    print('info =', info)
    if info != 0:
        print(f'WARNING: CG did not converge!')

    # Set up ac volume
    ac = ret.reshape((M,)*3)
    ac = np.ascontiguousarray(ac.real).astype(np.float64)
    print(f"Rank {comm.rank} got AC in {callback.counter} iterations.", flush=True)
    print('np.linalg.norm(ac) =', np.linalg.norm(ac))

    # Log autocorrelation volume
    image.show_volume(ac, settings.Mquat, f"autocorrelation_{generation}_{comm.rank}.png")

    return ac
