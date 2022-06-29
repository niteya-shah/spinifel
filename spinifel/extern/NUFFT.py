# Include some libraries that should be there for sure
from importlib.metadata import version
import numpy as np
import PyNVTX as nvtx
from spinifel import SpinifelSettings, SpinifelContexts, Profiler

import skopi as skp
from spinifel.extern import cufinufft_ext

settings = SpinifelSettings()
context  = SpinifelContexts()
profiler = Profiler()

if settings.use_cufinufft:
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    import pycuda.autoinit

    import cupy as cp
    from cufinufft import cufinufft
    mode = "cufinufft" + version("cufinufft") 

elif context.finufftpy_available:
    import finufft
    mode = "finufft" + version("finufft")

class NUFFT:
    def __init__(self,settings, pixel_position_reciprocal, pixel_distance_reciprocal) -> None:
        self.N_orientations = settings.N_orientations

        self.N_batch_size = settings.N_batch_size
        self.reduced_det_shape = settings.reduced_det_shape
        self.oversampling = settings.oversampling
        self.N_pixels = np.prod(self.reduced_det_shape)
        self.N_images = settings._N_images_per_rank

        self.pixel_position_reciprocal = pixel_position_reciprocal
        self.ref_orientations = skp.get_uniform_quat(self.N_orientations, True)
        ref_rotmat = np.array([np.linalg.inv(skp.quaternion2rot3d(quat)) for quat in self.ref_orientations])
        self.ref_rotmat = np.array(ref_rotmat)

        self.eps = 1e-12
        self.isign = -1


        self.reciprocal_extent = pixel_distance_reciprocal.max()
        self.pixel_position_reciprocal = np.array(pixel_position_reciprocal)
        self.mult = (np.pi / (self.oversampling * self.reciprocal_extent))

        if settings.use_cufinufft:
            self.H_f = gpuarray.empty(shape=(self.N_pixels * self.N_batch_size,), dtype=np.float64)
            self.K_f = gpuarray.empty(shape=(self.N_pixels * self.N_batch_size,), dtype=np.float64)
            self.L_f = gpuarray.empty(shape=(self.N_pixels * self.N_batch_size,), dtype=np.float64)

            self.H_a = gpuarray.empty(shape=(self.N_pixels * self.N_images,), dtype=np.float64)
            self.K_a = gpuarray.empty(shape=(self.N_pixels * self.N_images,), dtype=np.float64)
            self.L_a = gpuarray.empty(shape=(self.N_pixels * self.N_images,), dtype=np.float64)

            self.HKL_mat = pycuda.driver.pagelocked_empty((self.ref_rotmat.shape[1],self.ref_rotmat.shape[0],*pixel_position_reciprocal.shape[1:]), self.ref_rotmat.dtype)

        elif context.finufftpy_available:
            self.H_f = np.empty((self.N_pixels * self.N_batch_size,), dtype=np.float64)
            self.K_f = np.empty((self.N_pixels * self.N_batch_size,), dtype=np.float64)
            self.L_f = np.empty((self.N_pixels * self.N_batch_size,), dtype=np.float64)

            self.H_a = np.empty((self.N_pixels * self.N_images,), dtype=np.float64)
            self.K_a = np.empty((self.N_pixels * self.N_images,), dtype=np.float64)
            self.L_a = np.empty((self.N_pixels * self.N_images,), dtype=np.float64)

            self.HKL_mat = np.empty((self.ref_rotmat.shape[1],self.ref_rotmat.shape[0],*pixel_position_reciprocal.shape[1:]), self.ref_rotmat.dtype)
            
        np.einsum("ijk,klmn->jilmn", self.ref_rotmat, self.pixel_position_reciprocal, optimize='greedy', out=self.HKL_mat)
        self.HKL_mat *= self.mult
        assert np.max(np.abs(self.HKL_mat)) < 3*np.pi

    @staticmethod
    @nvtx.annotate("extern/util.py", is_prefix=True)
    def transpose(x, y, z):
        """Transposes the order of the (x, y, z) coordinates to (z, y, x)"""
        return z, y, x

    if settings.use_cuda:

        @staticmethod
        @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)
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

    if mode == "cufinufft1.2":
        @nvtx.annotate("NUFFT/cufinufft/forward", is_prefix=True)
        def forward(self, ugrid, st, en, support, use_recip_sym, N):
            H_ = self.HKL_mat[0,st:en,:].reshape(-1)
            K_ = self.HKL_mat[1,st:en,:].reshape(-1)
            L_ = self.HKL_mat[2,st:en,:].reshape(-1)

            assert H_.shape == K_.shape == L_.shape

            if use_recip_sym:
                assert (ugrid.real == ugrid).get().all()

            if support is not None:
                ugrid *= support 

            dev_id = cp.cuda.device.Device().id
            complex_dtype = np.complex128
            dtype         = np.float64

            H_, K_, L_ = self.transpose(H_, K_, L_)

            self.H_f.set(H_)
            self.K_f.set(K_)
            self.L_f.set(L_)

            nuvect = gpuarray.GPUArray(shape=(N,), dtype=complex_dtype)
            if not hasattr(self, "plan_f"):
                self.plan_f = cufinufft(2, ugrid.shape, 1, self.eps, isign=self.isign, dtype=dtype, gpu_method=1, gpu_device_id=dev_id)

            self.plan_f.set_pts(self.H_f, self.K_f, self.L_f)
            self.plan_f.execute(nuvect, ugrid)
            return self.gpuarray_to_cupy(nuvect)

        @nvtx.annotate("NUFFT/cufinufft/adjoint", is_prefix=True)
        def adjoint(self, nuvect, H_, K_, L_, support, use_reciprocal_symmetry, M):
            assert H_.shape == K_.shape == L_.shape
            dev_id = cp.cuda.device.Device().id
            complex_dtype = cp.complex128
            dtype = cp.float64

            H_, K_, L_ = self.transpose(H_, K_, L_)
            shape = (M, M, M)
    # TODO convert to GPUarray
            nuvect_ga = self.gpuarray_from_cupy(nuvect)

            ugrid = gpuarray.GPUArray(shape=shape, dtype=complex_dtype, order="F")
            self.H_a.set(H_)
            self.K_a.set(K_)
            self.L_a.set(L_)
            if not hasattr(self, "plan_a"):
                self.plan_a = {}
            if not shape in self.plan_a:
                self.plan_a[shape] = cufinufft(
                    1,
                    shape,
                    1,
                    self.eps,
                    isign=self.isign,
                    dtype=dtype,
                    gpu_method=1,
                    gpu_device_id=dev_id)
            self.plan_a[shape].set_pts(self.H_a, self.K_a, self.L_a)
            self.plan_a[shape].execute(nuvect_ga, ugrid)
            ugrid_gpu = self.gpuarray_to_cupy(ugrid)
            ugrid_gpu *= support
            if use_reciprocal_symmetry:
                ugrid_gpu = ugrid_gpu.real
            ugrid_gpu /= M**3
            return ugrid_gpu

    elif mode == "cufinufft1.1":
        def forward(self, ugrid, st, en, support, use_recip_sym, N):
            H_ = self.HKL_mat[0,st:en,:].reshape(-1)
            K_ = self.HKL_mat[1,st:en,:].reshape(-1)
            L_ = self.HKL_mat[2,st:en,:].reshape(-1)

            assert H_.shape == K_.shape == L_.shape

            if use_recip_sym:
                assert (ugrid.real == ugrid).all().get()

            if support is not None:
                ugrid *= support 

            dim = 3
            dev_id = cp.cuda.device.Device().id
            complex_dtype = np.complex128
            dtype         = np.float64

            H_, K_, L_ = self.transpose(H_, K_, L_)

            self.H_.set(H_)
            self.K_.set(K_)
            self.L_.set(L_)

            forward_opts = cufinufft.default_opts(nufft_type=2, dim=dim)
            forward_opts.gpu_method = 1   # Override with method 1. The default is 2
            forward_opts.cuda_device_id = dev_id

            nuvect = gpuarray.GPUArray(shape=(N,), dtype=complex_dtype)
    
            if not hasattr(self, "plan"):
                self.plan = cufinufft(2, ugrid.shape, 1, self.isign, self.eps, dtype=dtype, opts=forward_opts)

            self.plan.set_pts(self.H_f.shape[0], self.H_f, self.K_f, self.L_f)
            self.plan.execute(nuvect, ugrid)
            return self.gpuarray_to_cupy(nuvect)


        @nvtx.annotate("NUFFT/cufinufft/adjoint", is_prefix=True)
        def adjoint(self, nuvect, H_, K_, L_, support, use_reciprocal_symmetry, M):
            
            dim = 3
            assert H_.shape == K_.shape == L_.shape
            dev_id = cp.cuda.device.Device().id
            complex_dtype = cp.complex128
            dtype = cp.float64

            H_, K_, L_ = self.transpose(H_, K_, L_)
            shape = (M, M, M)
    # TODO convert to GPUarray
            nuvect_ga = self.gpuarray_from_cupy(nuvect)

            ugrid = gpuarray.GPUArray(shape=shape, dtype=complex_dtype, order="F")
            self.H_a.set(H_)
            self.K_a.set(K_)
            self.L_a.set(L_)
            
            adjoint_opts = cufinufft.default_opts(nufft_type=1, dim=dim)
            adjoint_opts.gpu_method = 1   # Override with method 1. The default is 2
            adjoint_opts.cuda_device_id = dev_id


            if not hasattr(self, "plan"):
                self.plan = {}
            if not shape in self.plan:
                self.plan[shape] = cufinufft(1, shape, self.isign, self.eps, dtype=dtype, opts=adjoint_opts)
            self.plan[shape].set_pts(self.H_, self.K_, self.L_)
            self.plan[shape].execute(nuvect_ga, ugrid)
            ugrid_gpu = self.gpuarray_to_cupy(ugrid)
            ugrid_gpu *= support
            if use_reciprocal_symmetry:
                ugrid_gpu = ugrid_gpu.real
            ugrid_gpu /= M**3
            return self.gpuarray_to_cupy(ugrid_gpu)

    elif mode == "finufft2.1.0":

        @nvtx.annotate("NUFFT/finufft/forward", is_prefix=True)
        def forward(self, ugrid, st, en, support, use_recip_sym, N):
            H_ = self.HKL_mat[0,st:en,:].reshape(-1)
            K_ = self.HKL_mat[1,st:en,:].reshape(-1)
            L_ = self.HKL_mat[2,st:en,:].reshape(-1)
            assert H_.shape == K_.shape == L_.shape

            # Allocate space in memory
            nuvect = np.zeros(N, dtype=np.complex128)

            #__________________________________________________________________________
            # Solve the NUFFT
            #

            assert not finufft.nufft3d2(H_, K_, L_, nuvect, self.isign, self.eps, ugrid)

            return nuvect


        @nvtx.annotate("NUFFT/finufft/adjoint", is_prefix=True)
        def adjoint(self, nuvect, H_, K_, L_, support, use_reciprocal_symmetry, M):
            """
            Version 1 of fiNUFFT 3D type 1
            """

            # Ensure that H, K, and L have the same shape
            assert H_.shape == K_.shape == L_.shape

            ugrid = np.zeros((M,M,M), dtype=np.complex128, order='F')

            #__________________________________________________________________________
            # Solve the NUFFT
            #

            finufft.nufft3d1(H_, K_, L_,nuvect.get(),n_modes=(M, M, M),isign=self.isign,eps= self.eps ,out= ugrid)

            #
            #--------------------------------------------------------------------------

            return ugrid

