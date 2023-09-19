import PyNVTX as nvtx

from spinifel.sequential.orientation_matching import (
    slicing_and_match as sequential_match,
)

from spinifel.sequential.orientation_matching import SNM
from spinifel.extern.orientations_ext import halo_generator
from spinifel import settings, contexts, Profiler
from concurrent.futures import ThreadPoolExecutor, as_completed

import time
import numpy as np

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

    from spinifel.extern.orientations_ext import TransferBufferGPU as TransferBuffer
else:
    from spinifel.extern.orientations_ext import TransferBufferCPU as TransferBuffer


# TODO: Make dtype int64? or controllable by precision?
if settings.use_single_prec:
    f_type = xp.float32
    c_type = xp.complex64
    int_type = xp.int32
else:
    f_type = xp.float64
    c_type = xp.complex128
    int_type = xp.int64


class SNM_MPI(SNM):
    def __init__(
        self,
        settings,
        slices_,
        pixel_position_reciprocal,
        pixel_distance_reciprocal,
        nufft,
    ):
        super().__init__(
            settings,
            slices_,
            pixel_position_reciprocal,
            pixel_distance_reciprocal,
            nufft,
        )
        self.N_batch = self.N_pixels * self.N_batch_size

    @nvtx.annotate("mpi/orientation_matching.py::modified", is_prefix=True)
    def _slice_and_match_int(self, stream_id, ugrid):
        transfer_buf = TransferBuffer((3, self.N_batch_size, self.N_pixels), f_type)
        contexts.ctx.push()

        slices_time = 0
        match_time = 0
        match_oth_time = 0
        slice_init = time.monotonic()


        for target_rank in rank_generator(contexts.size_compute_shared, contexts.size_compute, contexts.rank, stream_id, settings.N_streams, self.nufft.HKL_mat.splits):
            shared = target_rank // contexts.size_compute_shared == contexts.rank // contexts.size_compute_shared

            self.nufft.HKL_mat.lock(target_rank)
            if shared:
                target_arr = self.nufft.HKL_mat.get_win_local(target_rank %  contexts.size_compute_shared)

            for offset in range(self.nufft.HKL_mat.rank_shape[1]//self.N_batch_size):
                slice_start = time.monotonic()
                if shared:
                    # H, K and L
                    for dim in range(3):
                        arr_offset = (dim, slice(offset * self.N_batch_size, (offset + 1) * self.N_batch_size))
                        transfer_buf.set_data_local(target_arr[arr_offset].reshape(self.N_batch_size, self.N_pixels), dim)
                else:
                    for dim in range(3):
                        arr_offset_begin = self.nufft.HKL_mat.get_strides()[0:2] @ np.array([dim, offset * self.N_batch_size])
                        target = (arr_offset_begin, self.N_batch)
                        self.nufft.HKL_mat.get_win(target_rank, target, transfer_buf.cpu_buf[dim])

                    self.nufft.HKL_mat.flush(target_rank)
                    transfer_buf.set_data()
                
                H_, K_, L_ = map(self.nufft.gpuarray_from_cupy, transfer_buf.get_HKL())

                forward_result = self.nufft.forward(ugrid, H_, K_, L_, stream_id, 1, self.reciprocal_extent, self.N_batch)
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

                self.euclidean_dist(
                    data_images, self.slices_, self.slices_2, self.dist[stream_id, :-1])

                args_temp = self.dist[stream_id].argmin(axis=0)
                matching_indexes = args_temp != self.N_batch_size
                self.args[stream_id, matching_indexes] = (target_rank % (contexts.size_compute//self.nufft.HKL_mat.splits)) * self.nufft.HKL_mat.rank_shape[1] + offset * self.N_batch_size + args_temp[matching_indexes]
                min_distance = xp.take_along_axis(self.dist[stream_id], args_temp[None, :], 0).reshape(-1)
                self.dist[stream_id, -1] = min_distance
                match_time += time.monotonic() - match_middle
            
            self.nufft.HKL_mat.unlock(target_rank)
        contexts.ctx.pop()

    @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)
    def slicing_and_match_with_min_dist(self, ac):
        st_init = time.monotonic()
        if not self.N_slices:
            return xp.zeros((0, 4))

        if not hasattr(self, "dist"):
            self.dist = xp.full((settings.N_streams, self.N_batch_size + 1, self.N_slices), xp.finfo(xp.float64).max)
            self.args = xp.zeros((settings.N_streams, self.N_slices), dtype=int_type)
        if settings.use_cufinufft:
            ugrid = to_gpu(ac.astype(c_type))
        else:
            ugrid = ac.astype(c_type)

        if settings.N_streams == 1:
            self._slice_and_match_int(0, ugrid)
        else:
            futures = []
            with ThreadPoolExecutor(max_workers=settings.N_streams) as executor:
                for i in range(settings.N_streams):
                    futures.append(executor.submit(self._slice_and_match_int(i, ugrid)))           

            for future in as_completed(futures):
                if future.exception():
                    logger.log(repr(future.exception()), level=1)

        args_final = xp.take_along_axis(self.args, self.dist[:, self.N_batch_size].argmin(axis=0)[None, :], 0).get()
        distances_final = self.dist[:, self.N_batch_size].min(axis=0).get()
        return xp.squeeze(self.nufft.ref_orientations[args_final]), distances_final

    @nvtx.annotate("sequential/orientation_matching.py::modified", is_prefix=True)
    def slicing_and_match(self, ac):
        orients, dist = self.slicing_and_match_with_min_dist(ac)
        return orients

@nvtx.annotate("mpi/orientation_matching.py", is_prefix=True)
def match(
    ac,
    slices_,
    pixel_position_reciprocal,
    pixel_distance_reciprocal,
    ref_orientations=None,
):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    return sequential_match(
        ac,
        slices_,
        pixel_position_reciprocal,
        pixel_distance_reciprocal,
        ref_orientations=ref_orientations,
    )
