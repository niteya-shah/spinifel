# Include some libraries that should be there for sure
# NUFFT with the different orientations distributed over the nodes
from importlib.metadata import version
import numpy as np
import PyNVTX as nvtx
from spinifel import settings, contexts, Profiler

import skopi as skp
from spinifel.extern import cufinufft_ext

profiler = Profiler()


if settings.use_cufinufft:
    if settings.use_pygpu:
        import PybindGPU  as gpuarray
    else:
        import pycuda.gpuarray as gpuarray
        import pycuda

    import cupy as cp
    from cufinufft import cufinufft

    mode = "cufinufft" + version("cufinufft")

elif context.finufftpy_available:
    from . import nfft as finufft

    mode = "finufft" + version("finufftpy")
    if settings.use_cupy:
        import cupy as cp

from spinifel.extern.orientations_ext import SharedMemory, WindowManager

if settings.use_single_prec:
    f_type = np.float32
    c_type = np.complex64
else:
    f_type = np.float64
    c_type = np.complex128


class NUFFT_MPI:
    def __init__(
        self,
        settings,
        pixel_position_reciprocal,
        pixel_distance_reciprocal,
        images_per_rank=None,
        orientations=None,
    ) -> None:
        self.N_orientations = settings.N_orientations

        self.N_batch_size = settings.N_batch_size
        self.reduced_det_shape = settings.reduced_det_shape
        self.oversampling = settings.oversampling
        self.N_pixels = np.prod(self.reduced_det_shape)

        # For psana2 streaming, no. of images can grow over no. of generations
        if images_per_rank is None:
            self.N_images = settings.N_images_per_rank
        else:
            self.N_images = images_per_rank

        self.pixel_position_reciprocal = pixel_position_reciprocal
        if orientations is None:
            self.ref_orientations = skp.get_uniform_quat(self.N_orientations, True)
        else:
            self.ref_orientations = orientations

        self.work_unit_shared = self.N_orientations//contexts.size_compute_shared
        self.work_unit = self.N_orientations//contexts.size_compute
        # Save reference rotation matrix so that we dont re-create it every
        # time
        self.ref_rotmat = SharedMemory((self.N_orientations, 3, 3), f_type, split=False, pinned=False)

        for quat in range(self.work_unit_shared * contexts.rank_shared, self.work_unit_shared * (contexts.rank_shared + 1)):
            self.ref_rotmat[quat] = np.linalg.inv(skp.quaternion2rot3d(self.ref_orientations[quat]))

        self.eps = 1e-12
        self.isign = -1

        self.reciprocal_extent = pixel_distance_reciprocal.max()
        self.pixel_position_reciprocal = np.array(pixel_position_reciprocal)
        self.mult = np.pi / (self.oversampling * self.reciprocal_extent)
        
        if settings.use_cufinufft:
            # Store resused datastructures in memory so that we don't
            # constantly deallocate and realloate them
            # Transfer Buffer stores pinned memory for transfer and GPU memory in one 
            # boxed class
            # We use H K L in contiguous memory
            self.H_a = gpuarray.GPUArray(
                shape=(self.N_pixels * self.N_images,), dtype=f_type
            )
            self.K_a = gpuarray.GPUArray(
                shape=(self.N_pixels * self.N_images,), dtype=f_type
            )
            self.L_a = gpuarray.GPUArray(
                shape=(self.N_pixels * self.N_images,), dtype=f_type
            )

            # Store memory that we have to send to the gpu constantly in pinned
            # memory.
            self.HKL_mat = WindowManager(
                (              
                    self.ref_rotmat.shape[1],
                    self.ref_rotmat.shape[0],
                    *pixel_position_reciprocal.shape[1:],
                ), 
                f_type
            )
          
        elif contexts.finufftpy_available:
            # Shift finufft to Shared version
            # self.H_f = np.empty((self.N_pixels * self.N_batch_size,), dtype=f_type)
            # self.K_f = np.empty((self.N_pixels * self.N_batch_size,), dtype=f_type)
            # self.L_f = np.empty((self.N_pixels * self.N_batch_size,), dtype=f_type)

            self.H_a = np.empty((self.N_pixels * self.N_images,), dtype=f_type)
            self.K_a = np.empty((self.N_pixels * self.N_images,), dtype=f_type)
            self.L_a = np.empty((self.N_pixels * self.N_images,), dtype=f_type)

            self.HKL_mat = WindowManager(
                (              
                    self.ref_rotmat.shape[1],
                    self.ref_rotmat.shape[0],
                    *pixel_position_reciprocal.shape[1:],
                ), 
                f_type,
                pinned=False
            )

        # Cupy Einsum leaks memory so we dont use it
        # This version assumes that Orientations are split across all nodes as in Window Manager
        # TODO: Work out a way to link those two
        self.local_HKL_mat = self.HKL_mat.get_win_local(contexts.rank_shared)
        np.einsum(
            "ijk,klmn->jilmn",
            self.ref_rotmat[self.work_unit * contexts.rank:self.work_unit * (contexts.rank + 1)],
            self.pixel_position_reciprocal,
            optimize="greedy",
            dtype=f_type,
            out=self.local_HKL_mat,
        )
        self.local_HKL_mat *= self.mult
        assert np.max(np.abs(self.local_HKL_mat)) < 3 * np.pi

    @nvtx.annotate("extern/util.py", is_prefix=True)
    def update_fields(self, n_images_per_rank):
        if self.N_images == n_images_per_rank:  # nothing to update
            return
        n_images_old = self.N_images
        self.N_images = n_images_per_rank
        if settings.use_cufinufft:
            # force deletion of H_a, K_a, L_a if they haven't already been deleted
            if not n_images_old == 0 and not settings.use_pygpu:
                self.H_a.gpudata.free()
                self.K_a.gpudata.free()
                self.L_a.gpudata.free()
            # Store reused datastructures in memory so that we don't
            # constantly deallocate and realloate them
            self.H_a = gpuarray.GPUArray(
                shape=(self.N_pixels * self.N_images,), dtype=f_type
            )
            self.K_a = gpuarray.GPUArray(
                shape=(self.N_pixels * self.N_images,), dtype=f_type
            )
            self.L_a = gpuarray.GPUArray(
                shape=(self.N_pixels * self.N_images,), dtype=f_type
            )
        elif context.finufftpy_available:
            self.H_a = np.empty((self.N_pixels * self.N_images,), dtype=f_type)
            self.K_a = np.empty((self.N_pixels * self.N_images,), dtype=f_type)
            self.L_a = np.empty((self.N_pixels * self.N_images,), dtype=f_type)

    @staticmethod
    @nvtx.annotate("extern/util.py", is_prefix=True)
    def transpose(x, y, z, dtype=None):
        """Transposes the order of the (x, y, z) coordinates to (z, y, x)"""
        if dtype:
            return z.astype(dtype), y.astype(dtype), x.astype(dtype)
        return z, y, x

    if settings.use_cuda:

        @staticmethod
        @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)
        def gpuarray_to_cupy(arr):
            """
            Convert GPUarray(pycuda or PybindGPU) to cupy. The conversion is zero-cost.
            :param arr
            :return arr
            """
            assert isinstance(arr, gpuarray.GPUArray)
            if settings.use_pygpu:
                # Create a memory chunk from raw pointer and its size.
                mem = cp.cuda.UnownedMemory(arr.ptr, arr.nbytes, owner=arr)

                # Wrap it as a MemoryPointer.
                memptr = cp.cuda.MemoryPointer(mem, offset=0)

                # Create an ndarray view backed by the memory pointer.
                return cp.ndarray(arr.shape, dtype=arr.dtype, memptr=memptr, strides=arr.strides)
            else:
                return cp.asarray(arr)

        @staticmethod
        @nvtx.annotate("sequential/autocorrelation.py::modified", is_prefix=True)
        def gpuarray_from_cupy(arr):
            """
            Convert from cupy to GPUarray(pycuda). The conversion is zero-cost.
            :param arr
            :return arr
            """
            assert isinstance(arr, cp.ndarray)
            shape = arr.shape
            arr_dtype = arr.dtype

            def alloc(x):
                return arr.data.ptr

            if arr.flags.c_contiguous:
                order = "C"
            elif arr.flags.f_contiguous:
                order = "F"
            else:
                raise ValueError("arr order cannot be determined")

            if settings.use_pygpu:
                return gpuarray.GPUArray(allocator=gpuarray.Allocator(arr), order=order) 
            else:
                return gpuarray.GPUArray(
                    shape=shape, dtype=arr_dtype, allocator=alloc, order=order
                )

        def free_gpuarrays_and_cufinufft_plans(self):
            if not settings.use_pygpu:

                # TODO: Fix to remove Transfer Buffers and Plans
                # self.H_f.gpudata.free()
                # self.K_f.gpudata.free()
                # self.L_f.gpudata.free()
                self.H_a.gpudata.free()
                self.K_a.gpudata.free()
                self.L_a.gpudata.free()
            if hasattr(self, "plan_f"):
                del self.plan_f
            if hasattr(self, "plan_a"):
                del self.plan_a
            if hasattr(self, "plan"):
                del self.plan

    if mode == "cufinufft1.2":

        @nvtx.annotate("NUFFT/cufinufft/forward", is_prefix=True)
        def forward(self, ugrid, H_, K_, L_, stream_id, support, use_recip_sym, N):
            """
            New version of forward now accepts gpuarray/pygpu arrays in H_, K_, L_ directly
            We also pass stream_id to ensure that every thread has a unique plan
            """
            # Use reshapes instead of flattens because flatten creates copies
            assert H_.shape == K_.shape == L_.shape

            if use_recip_sym:
                # TODO: We need to fix this when .real is availble in PybindGPU
                if not settings.use_pygpu:
                    assert (ugrid.real == ugrid).get().all()

            if support is not None:
                if settings.use_pygpu:
                    ugrid_cpu_arr = ugrid.get()
                    ugrid_cpu_arr *= support
                    ugrid.set(ugrid_cpu_arr)
                else:
                    ugrid *= support

            dev_id = cp.cuda.device.Device().id

            H_, K_, L_ = self.transpose(H_, K_, L_)

            nuvect = gpuarray.GPUArray(shape=(N,), dtype=c_type)
            if not hasattr(self, "plan_f"):
                self.plan_f = {}
            if stream_id not in self.plan_f:
                self.plan_f[stream_id] = cufinufft(
                    2,
                    ugrid.shape,
                    1,
                    self.eps,
                    isign=self.isign,
                    dtype=f_type,
                    gpu_method=1,
                    gpu_device_id=dev_id,
                )

            self.plan_f[stream_id].set_pts(H_, K_, L_)
            self.plan_f[stream_id].execute(nuvect, ugrid)
            return self.gpuarray_to_cupy(nuvect)

        @nvtx.annotate("NUFFT/cufinufft/adjoint", is_prefix=True)
        def adjoint(self, nuvect, H_, K_, L_, support, use_reciprocal_symmetry, M):
            assert H_.shape == K_.shape == L_.shape
            dev_id = cp.cuda.device.Device().id

            H_, K_, L_ = self.transpose(H_, K_, L_, dtype=f_type)
            shape = (M, M, M)
            # TODO convert to GPUarray
            nuvect_ga = self.gpuarray_from_cupy(nuvect)
            ugrid = gpuarray.GPUArray(shape=shape, dtype=c_type, order="F")
            self.H_a.set(H_)
            self.K_a.set(K_)
            self.L_a.set(L_)
            if not hasattr(self, "plan_a"):
                self.plan_a = {}
            if shape not in self.plan_a:
                self.plan_a[shape] = cufinufft(
                    1,
                    shape,
                    1,
                    self.eps,
                    isign=self.isign,
                    dtype=f_type,
                    gpu_method=1,
                    gpu_device_id=dev_id,
                )
            self.plan_a[shape].set_pts(self.H_a, self.K_a, self.L_a)
            self.plan_a[shape].execute(nuvect_ga, ugrid)
            ugrid_cp = self.gpuarray_to_cupy(ugrid)
            ugrid_cp *= support
            if use_reciprocal_symmetry:
                ugrid_cp = ugrid_cp.real
            ugrid_cp /= M**3
            return ugrid_cp

    elif mode == "cufinufft1.1":

        def forward(self, ugrid, st, en, support, use_recip_sym, N):
            H_ = self.HKL_mat[0, st:en, :].reshape(-1)
            K_ = self.HKL_mat[1, st:en, :].reshape(-1)
            L_ = self.HKL_mat[2, st:en, :].reshape(-1)

            assert H_.shape == K_.shape == L_.shape

            if use_recip_sym:
                assert (ugrid.real == ugrid).all().get()

            if support is not None:
                ugrid *= support

            dim = 3
            dev_id = cp.cuda.device.Device().id
            H_, K_, L_ = self.transpose(H_, K_, L_)

            self.H_.set(H_)
            self.K_.set(K_)
            self.L_.set(L_)

            forward_opts = cufinufft.default_opts(nufft_type=2, dim=dim)
            forward_opts.gpu_method = 1  # Override with method 1. The default is 2
            forward_opts.cuda_device_id = dev_id

            nuvect = gpuarray.GPUArray(shape=(N,), dtype=c_type)

            if not hasattr(self, "plan"):
                self.plan = cufinufft(
                    2,
                    ugrid.shape,
                    1,
                    self.isign,
                    self.eps,
                    dtype=f_type,
                    opts=forward_opts,
                )

            self.plan.set_pts(self.H_f.shape[0], self.H_f, self.K_f, self.L_f)
            self.plan.execute(nuvect, ugrid)
            return self.gpuarray_to_cupy(nuvect)

        @nvtx.annotate("NUFFT/cufinufft/adjoint", is_prefix=True)
        def adjoint(self, nuvect, H_, K_, L_, support, use_reciprocal_symmetry, M):

            dim = 3
            assert H_.shape == K_.shape == L_.shape
            dev_id = cp.cuda.device.Device().id

            H_, K_, L_ = self.transpose(H_, K_, L_)
            shape = (M, M, M)
            # TODO convert to GPUarray
            nuvect_ga = self.gpuarray_from_cupy(nuvect)

            ugrid = gpuarray.GPUArray(shape=shape, dtype=c_type, order="F")
            self.H_a.set(H_)
            self.K_a.set(K_)
            self.L_a.set(L_)

            adjoint_opts = cufinufft.default_opts(nufft_type=1, dim=dim)
            adjoint_opts.gpu_method = 1  # Override with method 1. The default is 2
            adjoint_opts.cuda_device_id = dev_id

            if not hasattr(self, "plan"):
                self.plan = {}
            if shape not in self.plan:
                self.plan[shape] = cufinufft(
                    1, shape, self.isign, self.eps, dtype=f_type, opts=adjoint_opts
                )
            self.plan[shape].set_pts(self.H_, self.K_, self.L_)
            self.plan[shape].execute(nuvect_ga, ugrid)
            ugrid_gpu = self.gpuarray_to_cupy(ugrid)
            ugrid_gpu *= support
            if use_reciprocal_symmetry:
                ugrid_gpu = ugrid_gpu.real
            ugrid_gpu /= M**3
            return ugrid_gpu

    elif mode == "finufft2.1.0":

        @nvtx.annotate("NUFFT/finufft/forward", is_prefix=True)
        def forward(self, ugrid, st, en, support, use_recip_sym, N):
            """
            Version 1 of fiNUFFT 3D type 2
            """

            H_ = self.HKL_mat[0, st:en, :].reshape(-1)
            K_ = self.HKL_mat[1, st:en, :].reshape(-1)
            L_ = self.HKL_mat[2, st:en, :].reshape(-1)
            assert H_.shape == K_.shape == L_.shape

            if use_recip_sym:
                assert xp.all(xp.isreal(ugrid))
            if support is not None:
                ugrid *= support

            # Allocate space in memory
            nuvect = np.zeros(N, dtype=c_type)

            # __________________________________________________________________
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

            ugrid = np.zeros((M, M, M), dtype=c_type, order="F")

            # __________________________________________________________________
            # Solve the NUFFT
            #
            if not isinstance(nuvect, np.ndarray):
                nuvect = nuvect.get()

            assert not finufft.nufft3d1(
                H_,
                K_,
                L_,
                nuvect,
                n_modes=(M, M, M),
                isign=self.isign,
                eps=self.eps,
                out=ugrid,
            )

            #
            # ------------------------------------------------------------------
            ugrid *= support
            if use_reciprocal_symmetry:
                ugrid = ugrid.real
            ugrid /= M**3

            if settings.use_cupy:
                ugrid = cp.array(ugrid)

            return ugrid

    elif mode == "finufft1.1.2":

        @nvtx.annotate("NUFFT/finufft/forward", is_prefix=True)
        def forward(self, ugrid, st, en, support, use_recip_sym, N):
            """
            Version 1 of fiNUFFT 3D type 2
            """
            H_ = self.HKL_mat[0, st:en, :].reshape(-1)
            K_ = self.HKL_mat[1, st:en, :].reshape(-1)
            L_ = self.HKL_mat[2, st:en, :].reshape(-1)
            assert H_.shape == K_.shape == L_.shape

            # Allocate space in memory
            nuvect = np.zeros(N, dtype=c_type)

            # __________________________________________________________________
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

            ugrid = np.zeros((M, M, M), dtype=c_type, order="F")

            # __________________________________________________________________
            # Solve the NUFFT
            #
            if not isinstance(nuvect, np.ndarray):
                nuvect = nuvect.get().astype(c_type)

            assert not finufft.nufft3d1(
                H_, K_, L_, nuvect, self.isign, self.eps, M, M, M, ugrid
            )

            #
            # ------------------------------------------------------------------
            if settings.use_cupy:
                ugrid = cp.array(ugrid)

            ugrid *= support
            if use_reciprocal_symmetry:
                ugrid = ugrid.real
            ugrid /= M**3

            return ugrid
