import time
import numpy  as np
import skopi  as skp
import PyNVTX  as nvtx
import logging

from   spinifel import settings, utils, autocorrelation
import spinifel.sequential.nearest_neighbor as nn
from   spinifel import utils, autocorrelation, SpinifelSettings

settings = SpinifelSettings()
if settings.use_cupy:
    import os
    os.environ['CUPY_ACCELERATORS'] = "cub"

    from pycuda import gpuarray

    from cupyx.scipy.sparse.linalg import LinearOperator, cg
    from cupy.linalg import norm
    import cupy as xp

    from cupy.cublas import gemm
else:
    from scipy.linalg        import norm
    from scipy.sparse.linalg import LinearOperator, cg
    from scipy.linalg.blas import dgemm as gemm
    xp = np

if settings.use_cufinufft:
    from pycuda import gpuarray

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
            nufft):
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
        assert self.N_orientations % self.N_batch_size == 0, "N_orientations must be divisible by N_batch_size"

        self.N_slices = slices_.shape[0]
        assert slices_.shape == (self.N_slices,) + self.reduced_det_shape

        self.reciprocal_extent = pixel_distance_reciprocal.max()
        self.pixel_position_reciprocal = pixel_position_reciprocal
        self.slices_ = xp.array(
            slices_.reshape(
                (self.N_slices,
                 self.N_pixels)),
            dtype=f_type)
        self.slices_2 = xp.square(self.slices_).sum(axis=1)
        self.slices_std = self.slices_.std()
        self.nufft = nufft

    @nvtx.annotate("sequential/orientation_matching.py::modified",
                   is_prefix=True)
    def euclidean_gemm(self, x, y, out):
        """
        Thin wrapper for the GEMM function to support both scipy blas routines and cublas gemm
        """
        if settings.use_cupy:
            return gemm(
                'N', 'T', x, y, out=out, alpha=-2.0, beta=1.0)
        else:
            # This will not overwrite out as we dont use fortran order for our
            # C.
            twoxy = gemm(-2.0, x, y, beta=1.0, trans_a=0, trans_b=1)
            np.add(out, twoxy, out=out)
            return out


    @nvtx.annotate("sequential/orientation_matching.py::modified",
                   is_prefix=True)
    def euclidean_dist(self, x, y, y_2, dist, start, end):
        """
        Computes the pair-wise euclidean distance betwee two image groups. This formulation relies on blas support from CUDA/ROCM.
        The computation relies on the fact that
        (x - y)^2 = ||x||^2 + ||y||^2  - 2<x,y>
        ||y||^2 can be pre-computed and stored.
        -2 <x,y> can be written in blas notation as dgemm.
        """
        x = xp.array(x)
        x_2 = xp.square(x).sum(axis=1)
        xp.add(x_2[:, xp.newaxis], y_2[xp.newaxis, :], out=dist[start:end])
        return self.euclidean_gemm(x, y, dist[start:end])

    @nvtx.annotate("sequential/orientation_matching.py::modified",
                   is_prefix=True)
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
            self.dist = xp.empty(
                (self.N_orientations, self.N_slices), dtype=f_type)
        if settings.use_cufinufft:
            ugrid = gpuarray.to_gpu(ac.astype(c_type))
        else:
            ugrid = ac.astype(c_type)
        slices_time = 0
        match_time = 0
        match_oth_time = 0
        slice_init = time.monotonic()
        for i in range(self.N_orientations // self.N_batch_size):
            slice_start = time.monotonic()
            st = i * self.N_batch_size
            en = st + self.N_batch_size
            N_batch = self.N_pixels * self.N_batch_size
            st_m = i * self.N_batch_size
            en_m = st_m + self.N_batch_size

            forward_result = self.nufft.forward(
                ugrid, st, en, 1, self.reciprocal_extent, N_batch)
            data_images = forward_result.real.reshape(self.N_batch_size, -1)
            slices_time += time.monotonic() - slice_start
            match_start = time.monotonic()
            if settings.use_cupy and not settings.use_cufinufft:
                data_images = xp.array(data_images)
            if not settings.use_cupy and settings.use_cufinufft:
                data_images = data_images.get()
            data_images *= self.slices_std / data_images.std()
            match_middle = time.monotonic()
            match_oth_time += match_middle - match_start
            self.euclidean_dist(
                data_images,
                self.slices_,
                self.slices_2,
                self.dist,
                st_m,
                en_m)
            match_time += time.monotonic() - match_middle

        match_start = time.monotonic()
        # We rely on cub with its faster tree reduce algorithm
        index = self.dist.argmin(axis=0)
        if not isinstance(index, np.ndarray):
            index = index.get()
        en_match = time.monotonic()
        match_time += en_match - match_start

        print(f"Match tot:{en_match-st_init:.2f}s. slice={slices_time:.2f}s. match={match_time:.2f}s. slice_oh={slice_init-st_init:.2f}s. match_oh={match_oth_time:.2f}s.")
        return self.nufft.ref_orientations[index]

@nvtx.annotate("sequential/orientation_matching.py", is_prefix=True)
def slicing_and_match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal, ref_orientations=None):
    """
    Determine orientations of the data images by minimizing the euclidean
    distance with the reference images computed by randomly slicing through
    the autocorrelation.

    :param ac: autocorrelation of the current electron density estimate
    :param slices_: data images
    :param pixel_position_reciprocal: pixel positions in reciprocal space
    :param pixel_distance_reciprocal: pixel distance in reciprocal space
    :param ref_orientations: optional parameter for unit tests
    :return ref_orientations: array of quaternions matched to slices_
    """
    st_init = time.monotonic()
    logger = logging.getLogger(__name__)
    Mquat = settings.Mquat
    M = 4 * Mquat + 1
    N_orientations = settings.N_orientations
    N_batch_size = settings.N_batch_size
    N_pixels = utils.prod(settings.reduced_det_shape)
    N_slices = slices_.shape[0]
    assert slices_.shape == (N_slices,) + settings.reduced_det_shape
    N = N_pixels * N_orientations

    if not N_slices:
        return np.zeros((0, 4))

    if ref_orientations is None:
        ref_orientations = skp.get_uniform_quat(N_orientations, True)
    else:
        print(f'Warning: {ref_orientations.shape[0]} referenced orientations were given (unit test).')
    ref_rotmat = np.array([np.linalg.inv(skp.quaternion2rot3d(quat)) for quat in ref_orientations])
    reciprocal_extent = pixel_distance_reciprocal.max()

    # Calulate Model Slices in batch
    assert N_orientations % N_batch_size == 0, "N_orientations must be divisible by N_batch_size"
    slices_ = slices_.reshape((N_slices, N_pixels))
    model_slices_new = np.zeros((N,))

    st_slice = time.monotonic()

    for i in range(N_orientations//N_batch_size):
        st = i * N_batch_size
        en = st + N_batch_size
        H, K, L = np.einsum("ijk,klmn->jilmn", ref_rotmat[st:en], pixel_position_reciprocal)
        H_ = H.flatten() / reciprocal_extent * np.pi / settings.oversampling
        K_ = K.flatten() / reciprocal_extent * np.pi / settings.oversampling
        L_ = L.flatten() / reciprocal_extent * np.pi / settings.oversampling
        N_batch = N_pixels * N_batch_size
        st_m = i * N_batch_size * N_pixels
        en_m = st_m + (N_batch_size * N_pixels)
        model_slices_new[st_m:en_m] = autocorrelation.forward(
                ac, H_, K_, L_, 1, reciprocal_extent, N_batch).real
    en_slice = time.monotonic()

    # Imaginary part ~ numerical error
    model_slices_new = model_slices_new.reshape((N_orientations, N_pixels))
    data_model_scaling_ratio = slices_.std() / model_slices_new.std()
    print(f"Data/Model std ratio: {data_model_scaling_ratio}.", flush=True)
    model_slices_new *= data_model_scaling_ratio

    # Calculate Euclidean distance in batch to avoid running out of GPU Memory
    st_match = time.monotonic()
    index = nn.nearest_neighbor(model_slices_new, slices_, N_batch_size)
    en_match = time.monotonic()

    print(f"Match tot:{en_match-st_init:.2f}s. slice={en_slice-st_slice:.2f}s. match={en_match-st_match:.2f}s. slice_oh={st_slice-st_init:.2f}s. match_oh={st_match-en_slice:.2f}s.")
    return ref_orientations[index]
