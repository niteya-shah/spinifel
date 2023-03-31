import skopi as skp
import PyNVTX as nvtx

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, SymLogNorm

import numpy as np
from scipy.ndimage import gaussian_filter
from spinifel import settings, utils, image, autocorrelation

from spinifel import SpinifelSettings

settings = SpinifelSettings()
logger = utils.Logger(True, settings)

if settings.use_cupy:
    import os

    os.environ["CUPY_ACCELERATORS"] = "cub"

    from pycuda import gpuarray

    from cupyx.scipy.sparse.linalg import LinearOperator, cg
    from cupy.linalg import norm
    import cupy as xp
else:
    from scipy.linalg import norm
    from scipy.sparse.linalg import LinearOperator, cg

    xp = np

if settings.use_single_prec:
    f_type = xp.float32
    c_type = xp.complex64
else:
    f_type = xp.float64
    c_type = xp.complex128


class Merge:
    def __init__(
        self,
        settings,
        slices_,
        pixel_position_reciprocal,
        pixel_distance_reciprocal,
        nufft,
    ):
        # We store variables in memory so that we don't end up recreating them
        # every time
        self.M = settings.M
        self.N_images = slices_.shape[0]
        self.N = np.prod(slices_.shape)
        self.reciprocal_extent = pixel_distance_reciprocal.max()
        self.use_reciprocal_symmetry = True

        self.maxiter = settings.solve_ac_maxiter
        self.oversampling = settings.oversampling
        self.M_ups = settings.M_ups

        self.sign = 1
        self.eps = 1e-12

        self.pixel_position_reciprocal = np.array(pixel_position_reciprocal)
        self.mult = np.pi / (self.reciprocal_extent * settings.oversampling)

        self.rlambda = 1 / self.N / 1000
        self.flambda = 1e3

        data = np.array(slices_.reshape(-1), dtype=f_type)
        weights = np.ones(self.N, dtype=f_type)
        self.nuvect_Db = xp.array((data * weights).astype(c_type))
        self.nuvect = xp.ones_like(data, dtype=c_type)

        def callback(xk):
            callback.counter += 1

        self.callback = callback
        self.callback.counter = 0

        reduced_det_shape = settings.reduced_det_shape
        self.N_pixels = np.prod(reduced_det_shape)

        lu = np.linspace(-np.pi, np.pi, self.M)
        Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing="ij")
        Qu_ = np.sqrt(Hu_**2 + Ku_**2 + Lu_**2)
        F_antisupport = Qu_ > np.pi / settings.oversampling
        assert np.all(F_antisupport == F_antisupport[::-1, :, :])
        assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
        assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
        assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])
        self.F_antisupport = xp.array(F_antisupport)

        self.nufft = nufft

    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def get_non_uniform_positions(self, orientations):
        if orientations.shape[0] > 0:
            rotmat = np.array(
                [np.linalg.inv(skp.quaternion2rot3d(quat)) for quat in orientations]
            )
        else:
            rotmat = np.zeros((0, 3, 3))
            logger.log(
                "WARNING: gen_nonuniform_positions got empty orientation - returning h,k,l for Null rotation"
            )

        # TODO : How to ensure we support all formats of pixel_position reciprocal
        # Current support shape is(3, N_panels, Dim_x, Dim_y)
        # We save the optimal path of einsum so that we are faster after the
        # first run.
        if not hasattr(self, "einsum_path"):
            self.einsum_path = np.einsum_path(
                "ijk,klmn->jilmn",
                rotmat,
                self.pixel_position_reciprocal,
                optimize="optimal",
            )[0]
        H, K, L = np.einsum(
            "ijk,klmn->jilmn",
            rotmat,
            self.pixel_position_reciprocal,
            optimize=self.einsum_path,
        )
        # shape->[N_images] x det_shape
        return H, K, L

    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def fourier_reg(self, uvect, support):
        ugrid = uvect.reshape((self.M,) * 3) * support
        if self.use_reciprocal_symmetry:
            assert xp.all(xp.isreal(ugrid))
        
        # Calculate fft of the shifted ugrid. 
        if settings.use_fftx:
            ugrid_shifted = np.fft.ifftshift(ugrid)
            print(f"calling fftn in fourier_reg in sequential/autocorrelation.py {ugrid_shifted.flags.c_contiguous=}")
            start_time = time.time()
            F_ugrid_np = xp.fft.fftn(xp.fft.ifftshift(ugrid))  # / M**3
            end_time = time.time()
            np_time = end_time - start_time
            start_time = time.time()
            F_ugrid = fftxp.fft.fftn(ugrid_shifted)
            end_time = time.time()
            fftxp_time = end_time - start_time
            fftxp.utils.print_diff(np, F_ugrid_np, F_ugrid,
                                   "F_ugrid")
            print(f"FULL TIME F_ugrid_fftx: np {np_time} fftxp {fftxp_time}")
            fftxp.utils.print_array_info(np, np.fft.ifftshift(ugrid), "DATA shifted ugrid")
        else:
            F_ugrid = xp.fft.fftn(xp.fft.ifftshift(ugrid))  # / M**3
        

        F_reg = F_ugrid * xp.fft.ifftshift(self.F_antisupport)
        
        start_time = time.time()
        reg_inversed = xp.fft.ifftn(F_reg)
        end_time = time.time()
        np_time = end_time - start_time
        if settings.use_fftx:
            print(f"ORDER ifftn F_reg is {F_reg.flags.c_contiguous}")
            start_time = time.time()
            reg_inversed_fftx = fftxp.fft.ifftn(F_reg)
            end_time = time.time()
            fftxp_time = end_time - start_time
            fftxp.utils.print_diff(np, reg_inversed, reg_inversed_fftx, "reg_inversed")
            print(f"FULL TIME reg_inversed: np {np_time} fftxp {fftxp_time}")
            fftxp.utils.print_array_info(np, F_reg, "DATA F_reg")
            reg_inversed = reg_inversed_fftx

        reg = xp.fft.fftshift(reg_inversed)
        uvect = (reg * support).reshape(-1)
        if self.use_reciprocal_symmetry:
            uvect = uvect.real

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
    ugrid_conv = autocorrelation.adjoint(
        np.ones_like(data), H_, K_, L_, 1, M_ups,
        reciprocal_extent, use_reciprocal_symmetry)
    print(f'calling fftn in setup_linops in sequential/autocorrelation.py')
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) / M**3

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = autocorrelation.core_problem_convolution_spinifel(
            uvect, M, F_ugrid_conv_, M_ups, ac_support, use_reciprocal_symmetry, True)
        uvect_FDF = autocorrelation.fourier_reg(
            uvect, ac_support, F_antisupport, M, use_reciprocal_symmetry)
        uvect = uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    def core_problem_convolution(self, uvect, F_ugrid_conv_, ac_support):
        if self.use_reciprocal_symmetry:
            assert xp.all(xp.isreal(uvect))
        ugrid = uvect.reshape((self.M,) * 3) * ac_support
        ugrid_ups = xp.zeros((self.M_ups,) * 3, dtype=uvect.dtype)
        ugrid_ups[: self.M, : self.M, : self.M] = ugrid
        # Convolution = Fourier multiplication
        F_ugrid_ups = xp.fft.fftn(xp.fft.ifftshift(ugrid_ups))
        F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
        ugrid_conv_out_ups = xp.fft.fftshift(xp.fft.ifftn(F_ugrid_conv_out_ups))
        # Downsample
        ugrid_conv_out = ugrid_conv_out_ups[: self.M, : self.M, : self.M]
        ugrid_conv_out *= ac_support
        if self.use_reciprocal_symmetry:
            # Both ugrid_conv and ugrid are real, so their convolution
            # should be real, but numerical errors accumulate in the
            # imaginary part.
            ugrid_conv_out = ugrid_conv_out.real
        return ugrid_conv_out.reshape(-1)

    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def setup_linops(self, H, K, L, ac_support, x0):
        # We reshape instead of flatten because that makes a copy.
        H_ = H.reshape(-1) * self.mult
        K_ = K.reshape(-1) * self.mult
        L_ = L.reshape(-1) * self.mult

        ugrid_conv = xp.array(
            self.nufft.adjoint(
                self.nuvect, H_, K_, L_, 1, self.use_reciprocal_symmetry, self.M_ups
            )
        )

        F_ugrid_conv_ = xp.fft.fftn(xp.fft.ifftshift(ugrid_conv)) / self.M**3

        def W_matvec(uvect):
            """Define W part of the W @ x = d problem."""
            uvect_ADA = self.core_problem_convolution(uvect, F_ugrid_conv_, ac_support)
            uvect_FDF = self.fourier_reg(uvect, ac_support)
            uvect = uvect_ADA + self.rlambda * uvect + self.flambda * uvect_FDF
            return uvect

        W = LinearOperator(
            dtype=c_type, shape=(self.M**3, self.M**3), matvec=W_matvec
        )

        uvect_ADb = xp.array(
            self.nufft.adjoint(
                self.nuvect_Db,
                H_,
                K_,
                L_,
                ac_support,
                self.use_reciprocal_symmetry,
                self.M,
            ).flatten()
        )
        d = uvect_ADb + self.rlambda * x0

        return W, d

    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
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
            ac_support = (ac_smoothed > 1e-12).astype(f_type)
            ac_estimate *= ac_support
        ac_estimate = xp.array(ac_estimate)
        ac_support = xp.array(ac_support)
        x0 = ac_estimate.reshape(-1)

        W, d = self.setup_linops(H, K, L, ac_support, x0)
        ret, info = cg(W, d, x0=x0, maxiter=self.maxiter, callback=self.callback)
        ac = ret.reshape((self.M,) * 3).get()
        if self.use_reciprocal_symmetry:
            assert np.all(np.isreal(ac))
        ac = ac.real
        it_number = self.callback.counter

        return ac
