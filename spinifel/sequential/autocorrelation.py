import matplotlib.pyplot as plt
import numpy             as np
import PyNVTX            as nvtx

from matplotlib.colors   import LogNorm
from scipy.ndimage       import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

import skopi as skp

from spinifel import settings, utils, image, autocorrelation



@nvtx.annotate("sequential/autocorrelation.py", is_prefix=True)
def setup_linops(H, K, L, data,
                 ac_support, weights, x0,
                 M, N, reciprocal_extent,
                 rlambda, flambda,
                 use_reciprocal_symmetry):
    """Define W and d parts of the W @ x = d problem.

    W = A_adj*Da*A + rl*I  + fl*F_adj*Df*F
    d = A_adj*Da*b + rl*x0 + 0

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
    ugrid_conv = autocorrelation.adjoint_spinifel(
        np.ones_like(data), H_, K_, L_, 1, M_ups,
        reciprocal_extent, use_reciprocal_symmetry)
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) / M**3

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = autocorrelation.core_problem_convolution(
            uvect, M, F_ugrid_conv_, M_ups, ac_support, use_reciprocal_symmetry)
        uvect_FDF = autocorrelation.fourier_reg(
            uvect, ac_support, F_antisupport, M, use_reciprocal_symmetry)
        uvect = uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    W = LinearOperator(
        dtype=np.complex64,
        shape=(M**3, M**3),
        matvec=W_matvec)

    nuvect_Db = data * weights
    uvect_ADb = autocorrelation.adjoint_spinifel(
        nuvect_Db, H_, K_, L_, ac_support, M,
        reciprocal_extent, use_reciprocal_symmetry
    ).flatten()
    d = uvect_ADb + rlambda*x0

    return W, d



@nvtx.annotate("sequential/autocorrelation.py", is_prefix=True)
def solve_ac(generation,
             pixel_position_reciprocal,
             pixel_distance_reciprocal,
             slices_,
             orientations=None,
             ac_estimate=None):
    M = settings.M
    N_images = slices_.shape[0]
    N = utils.prod(slices_.shape)
    reciprocal_extent = pixel_distance_reciprocal.max()
    use_reciprocal_symmetry = True

    if orientations is None:
        orientations = skp.get_random_quat(N_images)
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations, pixel_position_reciprocal)

    if ac_estimate is None:
        ac_support = np.ones((M,)*3)
        ac_estimate = np.zeros((M,)*3)
    else:
        ac_smoothed = gaussian_filter(ac_estimate, 0.5)
        ac_support = (ac_smoothed > 1e-12).astype(np.float)
        ac_estimate *= ac_support
    
    data = slices_.flatten()
    weights = np.ones(N)

    # regularization paramters
    rlambda = 1/N / 1000
    flambda = 1e3

    maxiter = settings.solve_ac_maxiter

    idx = np.abs(L) < reciprocal_extent * .01
    plt.scatter(H[idx], K[idx], c=slices_[idx], s=1, norm=LogNorm())
    plt.axis('equal')
    plt.colorbar()
    plt.savefig(settings.out_dir / f"star_{generation}.png")
    plt.cla()
    plt.clf()

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    x0 = ac_estimate.flatten()
    W, d = setup_linops(H, K, L, data,
                        ac_support, weights, x0,
                        M, N, reciprocal_extent,
                        rlambda, flambda,
                        use_reciprocal_symmetry)
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    ac = ret.reshape((M,)*3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac))
    ac = ac.real
    it_number = callback.counter

    print(f"Recovered AC in {it_number} iterations.", flush=True)
    image.show_volume(ac, settings.Mquat, f"autocorrelation_{generation}.png")

    return ac
