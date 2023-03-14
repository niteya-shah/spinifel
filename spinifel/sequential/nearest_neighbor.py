from importlib.util import find_spec
import os
import numpy as np
import PyNVTX as nvtx

from sklearn.metrics.pairwise import euclidean_distances
from spinifel import SpinifelSettings, SpinifelContexts, Logger


# ______________________________________________________________________________
# Load global settings, and contexts
#

context = SpinifelContexts()
settings = SpinifelSettings()
logger = Logger(True, settings)

rank = context.rank


# _________________________________________________________________________
# TRY to import the cuda nearest neighbor pybind11 module -- if it exists in
# the path and we enabled `use_cuda`


class CUKNNRequiredButNotFound(Exception):
    """
    Settings require CUDA implementation of KNN, but the module is unavailable
    """


if settings.use_single_prec:
    knn_string = "spinifel.sequential.pyCudaKNearestNeighbors_SP"
else:
    knn_string = "spinifel.sequential.pyCudaKNearestNeighbors_DP"

KNN_LOADER = find_spec(knn_string)
KNN_AVAILABLE = KNN_LOADER is not None

logger.log(f"pyCudaKNearestNeighbors is available: {KNN_AVAILABLE}", level=1)

if settings.use_cuda and KNN_AVAILABLE:
    if settings.use_single_prec:
        import spinifel.sequential.pyCudaKNearestNeighbors_SP as pyCu
    else:
        import spinifel.sequential.pyCudaKNearestNeighbors_DP as pyCu
elif settings.use_cuda and not KNN_AVAILABLE:
    raise CUKNNRequiredButNotFound


# -------------------------------------------------------------------------


@nvtx.annotate("sequential/nearest_neighbor.py", is_prefix=True)
def generate_weights(pixel_position_reciprocal, order=0):
    """
    Generate weights: 1/(pixel_position_reciprocal)^order,
    to counteract the decay of the SAXS pattern.
    :param pixel_position_reciprocal: reciprocal space position of each pixel
    :param order: power, uniform weights if zero
    :return weights: resolution-based weight of each pixel
    """
    s_magnitudes = (
        np.linalg.norm(pixel_position_reciprocal, axis=0) * 1e-10
    )  # convert to Angstrom
    weights = 1.0 / (s_magnitudes**order)
    weights /= np.sum(weights)

    return weights


@nvtx.annotate("sequential/nearest_neighbor.py", is_prefix=True)
def calc_eudist_gpu(model_slices, slices, deviceId):
    model_slices_flat = model_slices.flatten()
    slices_flat = slices.flatten()

    euDist = pyCu.cudaEuclideanDistance(
        model_slices_flat,
        slices_flat,
        model_slices.shape[0],
        slices.shape[0],
        slices.shape[1],
        deviceId,
    )
    return euDist


@nvtx.annotate("sequential/nearest_neighbor.py", is_prefix=True)
def calc_argmin_gpu(euDist, n_images, n_refs, n_pixels, deviceId):

    index = pyCu.cudaHeapSort(euDist, n_images, n_refs, n_pixels, 1, deviceId)
    return index


@nvtx.annotate("sequential/nearest_neighbor.py", is_prefix=True)
def nearest_neighbor(model_slices, slices, batch_size):

    # detector size (total pixels) should be >= 16 to use CUDA code
    if settings.use_cuda and slices.shape[1] >= 16:
        deviceId = context.dev_id
        logger.log(
            f"Using CUDA  to calculate Euclidean distance and heap sort (batch_size={batch_size})",
            level=1
        )
        logger.log(f"Rank {rank} using deviceId {deviceId}", level=1)

        # Calculate Euclidean distance in batch to avoid running out of GPU
        # Memory
        euDist = np.zeros((slices.shape[0], model_slices.shape[0]), dtype=slices.dtype)
        for i in range(model_slices.shape[0] // batch_size):
            st = i * batch_size
            en = st + batch_size
            euDist[:, st:en] = calc_eudist_gpu(
                model_slices[st:en], slices, deviceId
            ).reshape(slices.shape[0], batch_size)
        euDist = euDist.flatten()

        index = calc_argmin_gpu(
            euDist, slices.shape[0], model_slices.shape[0], slices.shape[1], deviceId
        )
    else:
        logger.log("Using sklearn Euclidean Distance and numpy argmin", level=1)
        euDist = euclidean_distances(model_slices, slices)
        index = np.argmin(euDist, axis=0)

        minDist = np.zeros(
            slices.shape[0],
        )
        for i in range(slices.shape[0]):
            minDist[i] = euDist[index[i], i]
        meanMinDist = np.mean(minDist)
        stdMinDist = np.std(minDist)
        if rank == 0:
            logger.log(f"OM mean, std: {meanMinDist}, {stdMinDist}")

    return index
