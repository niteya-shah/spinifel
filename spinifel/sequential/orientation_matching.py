import os
os.environ['CUPY_ACCELERATORS']="cub,cutensor"

import time
import numpy  as np
import skopi  as skp
import cupy as cp
from pycuda import gpuarray
import pycuda.autoinit

from cufinufft import cufinufft

import PyNVTX  as nvtx

class SNM:

    def __init__(self, settings, slices_, pixel_position_reciprocal, pixel_distance_reciprocal):
        """
        Initialise our Slicing and matching class, storing reused data in memory so that we 
        we dont have to continously recalculate them.
        :param settings
        :param slices_
        :param xp(optional)
        """

        self.N_orientations = settings.N_orientations
        self.N_batch_size = settings.N_batch_size
        self.reduced_det_shape = settings.reduced_det_shape
        self.oversampling = settings.oversampling
        self.N_pixels = np.prod(self.reduced_det_shape)
        self.N = self.N_pixels * self.N_orientations
        self.ref_orientations = skp.get_uniform_quat(self.N_orientations, True)
        ref_rotmat = np.array([np.linalg.inv(skp.quaternion2rot3d(quat)) for quat in self.ref_orientations])
        self.ref_rotmat = cp.array(ref_rotmat)
        assert self.N_orientations % self.N_batch_size == 0, "N_orientations must be divisible by N_batch_size"

        self.eps = 1e-12
        self.isign = -1

        self.N_slices = slices_.shape[0]
        assert slices_.shape == (self.N_slices,) + self.reduced_det_shape

        self.reciprocal_extent = pixel_distance_reciprocal.max()
        self.pixel_position_reciprocal_gpu = cp.array(pixel_position_reciprocal)
        self.mult = (np.pi / (self.oversampling * self.reciprocal_extent))
        self.slices_ = cp.array(slices_.reshape((self.N_slices, self.N_pixels)), dtype=cp.float64)
        self.slices_2 = cp.square(self.slices_).sum(axis=1)
        self.slices_std = self.slices_.std()

    @staticmethod
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
    def gpuarray_to_cupy(arr):
        """
        Convert from cupy to GPUarray(pycuda). The conversion is zero-cost.
        :param arr
        :return arr
        """
        assert isinstance(arr, gpuarray.GPUArray)
        return cp.asarray(arr)

    #TODO Static method or not?
    @staticmethod
    def euclidean_dist(x ,y, y_2, dist, start, end):
        """
        Computes the pair-wise euclidean distance betwee two image groups. This formulation relies on blas support from CUDA/ROCM.
        The computation relies on the fact that 
        (x - y)^2 = ||x||^2 + ||y||^2  - 2<x,y>
        ||y||^2 can be pre-computed and stored.
        -2 <x,y> can be written in blas notation. 
        """
        x_2 = cp.square(x).sum(axis=1)
        cp.add(x_2[:,cp.newaxis], y_2[cp.newaxis,:],out=dist[start:end])
        return cp.cublas.gemm('N','T',x,y,out=dist[start:end],alpha=-2.0,beta=1.0)

    @staticmethod
    def transpose(x, y, z):
        """Transposes the order of the (x, y, z) coordinates to (z, y, x)"""
        return z, y, x

    def forward(self, ugrid, H_, K_, L_, support, use_recip_sym, N):

        assert H_.shape == K_.shape == L_.shape
        # Use one of the stack functions or do piecwise? this will allocate memory for no reason
        assert cp.max(cp.abs(cp.array([H_, K_, L_]))) < 3*np.pi

        if use_recip_sym:
            assert cp.all(cp.isreal(ugrid))

        if support is not None:
            ugrid *= support 

        dev_id = cp.cuda.device.Device().id
        complex_dtype = cp.complex128
        dtype         = cp.float64

        H_, K_, L_ = self.transpose(H_, K_, L_)

        #TODO convert to GPUarray
        H_,K_,L_,ugrid = map(self.gpuarray_from_cupy, [H_,K_,L_,ugrid])

        nuvect_gpu = gpuarray.GPUArray(shape=(N,), dtype=complex_dtype)
        if not hasattr(self, "plan"):
            self.plan = cufinufft(2, ugrid.shape, 1, self.eps, isign=self.isign, dtype=dtype,gpu_method=1, gpu_device_id=dev_id)
        
        self.plan.set_pts(H_, K_, L_)
        self.plan.execute(nuvect_gpu, ugrid)

        return nuvect_gpu

    @nvtx.annotate("sequential/main.py::modified", is_prefix=True)
    def slicing_and_match(self, ac):
        """
        Determine orientations of the data images by minimizing the euclidean distance with the reference images 
        computed by randomly slicing through the autocorrelation.

        MONA: This is a current hack to support Legion. For MPI, slicing is done separately 
        from orientation matching. 

        NITEYA: This version shifts to cupy completely to speedup the computation

        :param ac: autocorrelation of the current electron density estimate
        :param slices_: data images
        :param pixel_position_reciprocal: pixel positions in reciprocal space
        :param pixel_distance_reciprocal: pixel distance in reciprocal space
        :return ref_orientations: array of quaternions matched to slices_
        """


        if not self.N_slices:
            return cp.zeros((0, 4))

        if not hasattr(self, "dist"):
            self.dist = cp.empty((self.N_orientations, self.N_slices))
        ugrid_gpu = cp.array(ac, dtype=cp.complex128)
        for i in range(self.N_orientations//self.N_batch_size):
            st = i * self.N_batch_size
            en = st + self.N_batch_size
            N_batch = self.N_pixels * self.N_batch_size
            st_m = i * self.N_batch_size
            en_m = st_m + self.N_batch_size

            #Question are H,K,L constant? Can we precalculate them and then store them?
            H, K, L = cp.einsum("ijk,klmn->jilmn", self.ref_rotmat[st:en], self.pixel_position_reciprocal_gpu,optimize=True) * self.mult
            H_ = H.flatten() 
            K_ = K.flatten()
            L_ = L.flatten()

            data_images = self.gpuarray_to_cupy(self.forward(ugrid_gpu, H_,K_,L_, 1, self.reciprocal_extent, N_batch)).real.reshape(self.N_batch_size, -1)
            data_images *= self.slices_std/data_images.std()

            #TODO Approximate data model scaling? Is it allowed?
            self.euclidean_dist(data_images, self.slices_, self.slices_2, self.dist, st_m, en_m)
        index = self.dist.argmin(axis=0).get()
        return self.ref_orientations[index]