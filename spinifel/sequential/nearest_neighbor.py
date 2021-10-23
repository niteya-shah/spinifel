import os
import numpy  as np
import PyNVTX as nvtx

from   sklearn.metrics.pairwise import euclidean_distances
from   spinifel                 import SpinifelSettings, SpinifelContexts



#______________________________________________________________________________
# Load global settings, and contexts
#

context = SpinifelContexts()
settings = SpinifelSettings()

rank = context.rank



#_______________________________________________________________________________
# TRY to import the cuda nearest neighbor pybind11 module -- if it exists in
# the path and we enabled `use_cuda`

from importlib.util import find_spec


class CUKNNRequiredButNotFound(Exception):
    """
    Settings require CUDA implementation of KNN, but the module is unavailable
    """



KNN_LOADER    = find_spec("spinifel.sequential.pyCudaKNearestNeighbors")
KNN_AVAILABLE = KNN_LOADER is not None
if settings.verbose:
    print(f"pyCudaKNearestNeighbors is available: {KNN_AVAILABLE}")

if settings.use_cuda and KNN_AVAILABLE:
    import spinifel.sequential.pyCudaKNearestNeighbors as pyCu
elif settings.use_cuda and not KNN_AVAILABLE:
    raise CUKNNRequiredButNotFound


#-------------------------------------------------------------------------------

@nvtx.annotate("sequential/nearest_neighbor.py", is_prefix=True)
def generate_weights(pixel_position_reciprocal, order=0):
    """
    Generate weights: 1/(pixel_position_reciprocal)^order,
    to counteract the decay of the SAXS pattern.
    :param pixel_position_reciprocal: reciprocal space position of each pixel
    :param order: power, uniform weights if zero
    :return weights: resolution-based weight of each pixel
    """
    s_magnitudes = np.linalg.norm(pixel_position_reciprocal, axis=0) * 1e-10 # convert to Angstrom
    weights = 1.0 / (s_magnitudes ** order)
    weights /= np.sum(weights)
    
    return weights


@nvtx.annotate("sequential/nearest_neighbor.py", is_prefix=True)
def calc_eudist_gpu(model_slices, slices, deviceId):
    model_slices_flat = model_slices.flatten()
    slices_flat       = slices.flatten()

    euDist = pyCu.cudaEuclideanDistance(slices_flat,
                                        model_slices_flat,
                                        slices.shape[0],
                                        model_slices.shape[0],
                                        slices.shape[1],
                                        deviceId)
    return euDist



@nvtx.annotate("sequential/nearest_neighbor.py", is_prefix=True)
def calc_argmin_gpu(euDist, n_images, n_refs, n_pixels, deviceId):

    index =  pyCu.cudaHeapSort(euDist,
                               n_images,
                               n_refs,
                               n_pixels,
                               1,
                               deviceId)
    return index



@nvtx.annotate("sequential/nearest_neighbor.py", is_prefix=True)
def nearest_neighbor(model_slices, slices, batch_size):

    if settings.use_cuda:
        deviceId = rank % settings._devices_per_node
        if settings.verbose:
            print(f"Using CUDA  to calculate Euclidean distance and heap sort (batch_size={batch_size})")
            print(f"Rank {rank} using deviceId {deviceId}")
        
        # Calculate Euclidean distance in batch to avoid running out of GPU Memory
        euDist = np.zeros((slices.shape[0], model_slices.shape[0]), dtype=slices.dtype)
        for i in range(model_slices.shape[0]//batch_size):
            st = i * batch_size
            en = st + batch_size
            euDist[:, st:en] = calc_eudist_gpu(model_slices[st:en], slices, deviceId).reshape(slices.shape[0], batch_size)
        euDist = euDist.flatten()
        
        index = calc_argmin_gpu(euDist, 
                            slices.shape[0],
                            model_slices.shape[0],
                            slices.shape[1],
                            deviceId)
    else:
        if settings.verbose:
            print("Using sklearn Euclidean Distance and numpy argmin")
        euDist    = euclidean_distances(model_slices, slices)
        index  = np.argmin(euDist, axis=0)

    return index

