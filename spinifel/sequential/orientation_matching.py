import time
import numpy as np
import skopi as skp
import PyNVTX as nvtx
import logging

from spinifel import settings, utils
import spinifel.sequential.nearest_neighbor as nn
from spinifel import utils, SpinifelSettings, Logger

settings = SpinifelSettings()
logger = Logger(True, settings)

if settings.use_cupy:
    import os

    os.environ["CUPY_ACCELERATORS"] = "cub"

    if settings.use_pygpu:
        from PybindGPU import to_gpu
    else:
        from pycuda.gpuarray import to_gpu

    import cupy as xp

    from cupy.cublas import gemm
    from cupyx.scipy.special import softmax
else:
    from scipy.linalg.blas import dgemm as gemm
    from scipy.special import softmax
    xp = np

if settings.use_cufinufft:
    if settings.use_pygpu:
        from PybindGPU import to_gpu
    else:
        from pycuda.gpuarray import to_gpu

if settings.use_single_prec:
    f_type = xp.float32
    c_type = xp.complex64
else:
    f_type = xp.float64
    c_type = xp.complex128


class SNM:
    def __init__(
        self,
        settings,
        slices_,
        pixel_position_reciprocal,
        pixel_distance_reciprocal,
        nufft,
    ):
        """
        Initialise our Slicing and matching class, storing reused data in memory so that we
        we dont have to continously recalculate them.
        :param settings
        :param slices_
        :param xp(optional)
        """
        # We save the datastructures we will use again
        self.N_orientations = settings.N_orientations
        self.N_batch_size = settings.N_batch_size
        self.reduced_det_shape = settings.reduced_det_shape
        self.oversampling = settings.oversampling
        self.N_pixels = np.prod(self.reduced_det_shape)
        self.N = self.N_pixels * self.N_orientations
        assert (
            self.N_orientations % self.N_batch_size == 0
        ), "N_orientations must be divisible by N_batch_size"

        self.N_slices = slices_.shape[0]
        assert slices_.shape == (self.N_slices,) + self.reduced_det_shape

        self.reciprocal_extent = pixel_distance_reciprocal.max()
        self.pixel_position_reciprocal = pixel_position_reciprocal
        self.slices_ = xp.array(
            slices_.reshape((self.N_slices, self.N_pixels)), dtype=f_type
        )
        
        self.slices_std = self.slices_.std()
        self.clip = settings.max_intensity_clip
        self.slices_ = self.intensity_clip(self.slices_, self.clip)
        
        self.slices_2 = xp.square(self.slices_).sum(axis=1)
        
        self.nufft = nufft

    @staticmethod
    @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)
    def intensity_clip(data, thresh):
        """
        clip pixel intensities above threshold, i.e. pix_val = min(pix_val, thresh)
        """
        if thresh > 0:
            ind = xp.where(data >= thresh)
            data[ind] = thresh
        return data
        
    @staticmethod
    @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)
    def euclidean_gemm(x, y, out):
        """
        Thin wrapper for the GEMM function to support both scipy blas routines and cublas gemm
        """
        if settings.use_cupy:
            return gemm("N", "T", x, y, out=out, alpha=-2.0, beta=1.0)
        else:
            # This will not overwrite out as we dont use fortran order for our
            # C.
            twoxy = gemm(-2.0, x, y, beta=1.0, trans_a=0, trans_b=1)
            np.add(out, twoxy, out=out)
            return out

    @staticmethod
    @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)
    def euclidean_dist(x, y, y_2, dist):
        """
        Computes the pair-wise euclidean distance betwee two image groups. This formulation relies on blas support from CUDA/ROCM.
        The computation relies on the fact that
        (x - y)^2 = ||x||^2 + ||y||^2  - 2<x,y>
        ||y||^2 can be pre-computed and stored.
        -2 <x,y> can be written in blas notation as dgemm.
        """
        x = xp.array(x)
        x_2 = xp.square(x).sum(axis=1)
        xp.add(x_2[:, xp.newaxis], y_2[xp.newaxis, :], out=dist)
        return SNM.euclidean_gemm(x, y, dist)

    @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)
    def slicing_and_match_with_min_dist(self, ac):

        orients = self.slicing_and_match(ac)
        mindist = self.dist.min(axis=0)
        if not isinstance(mindist, np.ndarray):
            mindist = mindist.get()
        return orients, mindist

    @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)

    # this is a place holder for conformation result based on
    # min distance values from each diffraction pattern
    # it needs to be updated
    def conformation_result(self, min_dist, mode):
        logger.log(f"conformation_result:mode = {mode}", level=2)
        if mode == "max_likelihood":
            min_d = xp.array(min_dist)
            min_v = xp.min(min_d,axis=0).reshape(1,-1)
            result = xp.where(min_d == min_v, 1.0, 0.0)
            if not isinstance(result, np.ndarray):
                result = result.get()
        elif mode == "softmax":
            result = softmax(xp.array(-min_dist), axis=0)
            if not isinstance(result, np.ndarray):
                result = result.get()
        # testing mode
        else:
            assert mode == "test_debug"
            result = np.ones(min_dist.shape)
        logger.log(f"conformation_result: result={result.shape}", level=2)
        return result

    @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)
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

        st_init = time.monotonic()
        if not self.N_slices:
            return xp.zeros((0, 4))

        if not hasattr(self, "dist"):
            self.dist = xp.empty((self.N_orientations, self.N_slices), dtype=f_type)
        if settings.use_cufinufft:
            ugrid = to_gpu(ac.astype(c_type))
        else:
            ugrid = ac.astype(c_type)
        slices_time = 0
        match_time = 0
        match_oth_time = 0
        slice_init = time.monotonic()

        # Note: [Niteya] Each rank calculates 10k orientation independently
        # and applies that to its own set of images.
        for i in range(self.N_orientations // self.N_batch_size):
            slice_start = time.monotonic()
            st = i * self.N_batch_size
            en = st + self.N_batch_size
            N_batch = self.N_pixels * self.N_batch_size
            st_m = i * self.N_batch_size
            en_m = st_m + self.N_batch_size

            forward_result = self.nufft.forward(
                ugrid, st, en, 1, self.reciprocal_extent, N_batch
            )
            data_images = forward_result.real.reshape(self.N_batch_size, -1)
            slices_time += time.monotonic() - slice_start
            match_start = time.monotonic()
            if settings.use_cupy and not settings.use_cufinufft:
                data_images = xp.array(data_images)
            if not settings.use_cupy and settings.use_cufinufft:
                data_images = data_images.get()
                
            data_images *= self.slices_std / data_images.std()
            
            data_images = self.intensity_clip(data_images, self.clip)
            
            match_middle = time.monotonic()
            match_oth_time += match_middle - match_start
            SNM.euclidean_dist(
                data_images, self.slices_, self.slices_2, self.dist[st_m:en_m])
            
            match_time += time.monotonic() - match_middle

        match_start = time.monotonic()
        # We rely on cub with its faster tree reduce algorithm
        index = self.dist.argmin(axis=0)
        if not isinstance(index, np.ndarray):
            index = index.get()
        en_match = time.monotonic()
        match_time += en_match - match_start

        logger.log(
            f"Match tot:{en_match-st_init:.2f}s. slice={slices_time:.2f}s. match={match_time:.2f}s. slice_oh={slice_init-st_init:.2f}s. match_oh={match_oth_time:.2f}s.",
            level=1
        )
        return self.nufft.ref_orientations[index]


