import matplotlib.pyplot as plt
import numpy             as np
import PyNVTX            as nvtx

from matplotlib.colors   import LogNorm
from scipy.ndimage       import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

import skopi as skp

from spinifel import settings, utils, image, autocorrelation


import os
os.environ['CUPY_ACCELERATORS'] = "cub,cutensor"

from pycuda import gpuarray
import pycuda.autoinit

import skopi as skp
import PyNVTX as nvtx

from cufinufft import cufinufft

from cupyx.scipy.sparse.linalg import LinearOperator, cg
import cupy as cp
from scipy.ndimage import gaussian_filter
import numpy as np

class Merge:

    def __init__(
            self,
            settings,
            slices_,
            pixel_position_reciprocal,
            pixel_distance_reciprocal):
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

        data = np.array(slices_.reshape(-1), dtype=cp.complex128)
        weights = np.ones(self.N)
        self.nuvect_Db = cp.array((data * weights).astype(np.complex128))
        self.nuvect = cp.ones_like(data, dtype=cp.complex128)

        def callback(xk):
            callback.counter += 1

        self.callback = callback
        self.callback.counter = 0

        reduced_det_shape = settings.reduced_det_shape
        self.N_pixels = np.prod(reduced_det_shape)

        self.H_ = gpuarray.empty(shape=(self.N_pixels * self.N_images,), dtype=np.float64)
        self.K_ = gpuarray.empty(shape=(self.N_pixels * self.N_images,), dtype=np.float64)
        self.L_ = gpuarray.empty(shape=(self.N_pixels * self.N_images,), dtype=np.float64)


        lu = np.linspace(-np.pi, np.pi, self.M)
        Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
        Qu_ = np.sqrt(Hu_**2 + Ku_**2 + Lu_**2)
        F_antisupport = Qu_ > np.pi / settings.oversampling
        assert np.all(F_antisupport == F_antisupport[::-1, :, :])
        assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
        assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
        assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])
        self.F_antisupport = cp.array(F_antisupport)


    @staticmethod
    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def gpuarray_from_cupy(arr):
        """
        Convert from GPUarray(pycuda) to cupy. The conversion is zero-cost.
        :param arr
        :return arr
        """
        assert isinstance(arr, cp.ndarray)
        shape = arr.shape
        dtype = arr.dtype

        def alloc(x):
            return arr.data.ptr

        if arr.flags.c_contiguous:
            order = 'C'
        elif arr.flags.f_contiguous:
            order = 'F'
        else:
            raise ValueError('arr order cannot be determined')
        return gpuarray.GPUArray(shape=shape,
                                 dtype=dtype,
                                 allocator=alloc,
                                 order=order)

    @staticmethod
    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def gpuarray_to_cupy(arr):
        """
        Convert from cupy to GPUarray(pycuda). The conversion is zero-cost.
        :param arr
        :return arr
        """
        assert isinstance(arr, gpuarray.GPUArray)
        return cp.asarray(arr)

    @staticmethod
    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def transpose(x, y, z):
        """Transposes the order of the (x, y, z) coordinates to (z, y, x)"""
        return z, y, x

    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def get_non_uniform_positions(self, orientations):
        if orientations.shape[0] > 0:
            rotmat = np.array([np.linalg.inv(skp.quaternion2rot3d(quat))
                              for quat in orientations])
        else:
            rotmat = np.zeros((0, 3, 3))
            print(
                "WARNING: gen_nonuniform_positions got empty orientation - returning h,k,l for Null rotation")

# TODO : How to ensure we support all formats of pixel_position reciprocal
# Current support shape is(3, N_panels, Dim_x, Dim_y)
        if not hasattr(self, "einsum_path"):
            self.einsum_path = np.einsum_path("ijk,klmn->jilmn", rotmat,
                            self.pixel_position_reciprocal, optimize="optimal")[0]
        H, K, L = np.einsum("ijk,klmn->jilmn", rotmat,
                            self.pixel_position_reciprocal, optimize=self.einsum_path) 
#H, K, L = np.einsum("ijk,klm->jilm", rotmat, pixel_position_reciprocal)
# shape->[N_images] x det_shape
        return H, K, L

    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def adjoint(self, nuvect, H_, K_, L_, support, M):
        assert H_.shape == K_.shape == L_.shape
        dev_id = cp.cuda.device.Device().id
        complex_dtype = cp.complex128
        dtype = cp.float64

        H_, K_, L_ = self.transpose(H_, K_, L_)
        shape = (M, M, M)
# TODO convert to GPUarray
        nuvect_ga = self.gpuarray_from_cupy(nuvect)

        ugrid = gpuarray.GPUArray(shape=shape, dtype=complex_dtype, order="F")
        self.H_.set(H_)
        self.K_.set(K_)
        self.L_.set(L_)
        if not hasattr(self, "plan"):
            self.plan = {}
        if not shape in self.plan:
            self.plan[shape] = cufinufft(
                1,
                shape,
                1,
                self.eps,
                isign=self.sign,
                dtype=dtype,
                gpu_method=1,
                gpu_device_id=dev_id)
        self.plan[shape].set_pts(self.H_, self.K_, self.L_)
        self.plan[shape].execute(nuvect_ga, ugrid)
        ugrid_gpu = self.gpuarray_to_cupy(ugrid)
        ugrid_gpu *= support
        if self.use_reciprocal_symmetry:
            ugrid_gpu = ugrid_gpu.real
        ugrid_gpu /= M**3
        return ugrid_gpu

    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def fourier_reg(self, uvect, support):
        ugrid = uvect.reshape((self.M,) * 3) * support
        if self.use_reciprocal_symmetry:
            assert cp.all(cp.isreal(ugrid))

        F_ugrid = cp.fft.fftn(cp.fft.ifftshift(ugrid))  # / M**3
        F_reg = F_ugrid * cp.fft.ifftshift(self.F_antisupport)
        reg = cp.fft.fftshift(cp.fft.ifftn(F_reg))
        uvect = (reg * support).reshape(-1)
        if self.use_reciprocal_symmetry:
            uvect = uvect.real

        return uvect

    def core_problem_convolution(self, uvect, F_ugrid_conv_, ac_support):
        if self.use_reciprocal_symmetry:
            assert cp.all(cp.isreal(uvect))
        ugrid = uvect.reshape((self.M,) * 3) * ac_support
        ugrid_ups = cp.zeros((self.M_ups,) * 3, dtype=uvect.dtype)
        ugrid_ups[:self.M, :self.M, :self.M] = ugrid
        # Convolution = Fourier multiplication
        F_ugrid_ups = cp.fft.fftn(cp.fft.ifftshift(ugrid_ups))
        F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
        ugrid_conv_out_ups = cp.fft.fftshift(
            cp.fft.ifftn(F_ugrid_conv_out_ups))
        # Downsample
        ugrid_conv_out = ugrid_conv_out_ups[:self.M, :self.M, :self.M]
        ugrid_conv_out *= ac_support
        if self.use_reciprocal_symmetry:
            # Both ugrid_conv and ugrid are real, so their convolution
            # should be real, but numerical errors accumulate in the
            # imaginary part.
            ugrid_conv_out = ugrid_conv_out.real
        return ugrid_conv_out.reshape(-1)

    @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
    def setup_linops(self, H, K, L, ac_support, x0):
        H_ = H.reshape(-1) * self.mult
        K_ = K.reshape(-1) * self.mult
        L_ = L.reshape(-1) * self.mult

        ugrid_conv = self.adjoint(self.nuvect, H_, K_, L_, 1, self.M_ups)

        F_ugrid_conv_ = cp.fft.fftn(
            cp.fft.ifftshift(ugrid_conv))/self.M**3

        def W_matvec(uvect):
            """Define W part of the W @ x = d problem."""
            uvect_ADA = self.core_problem_convolution(
                uvect, F_ugrid_conv_, ac_support)
            uvect_FDF = self.fourier_reg(uvect, ac_support)
            uvect = uvect_ADA + self.rlambda * uvect + self.flambda * uvect_FDF
            return uvect

        W = LinearOperator(
            dtype=np.complex128,
            shape=(self.M**3, self.M**3),
            matvec=W_matvec)
    
        uvect_ADb = self.adjoint(
            self.nuvect_Db, H_, K_, L_, ac_support, self.M).flatten()
        d = uvect_ADb + self.rlambda * x0

        return W, d

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
        ac_estimate = cp.array(ac_estimate)
        ac_support = cp.array(ac_support)
        x0 = ac_estimate.reshape(-1)

        W, d = self.setup_linops(H, K, L, ac_support, x0)
        ret, info = cg(W, d, x0=x0, maxiter=self.maxiter,
                       callback=self.callback)
        ac = ret.reshape((self.M,) * 3).get()
        if self.use_reciprocal_symmetry:
            assert np.all(np.isreal(ac))
        ac = ac.real
        it_number = self.callback.counter

        return ac
