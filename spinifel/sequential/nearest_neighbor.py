import os
import numpy as np
from   sklearn.metrics.pairwise import euclidean_distances
from   spinifel                 import SpinifelSettings

# TODO: This should be included in spinifel contexts
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()


settings = SpinifelSettings()

#_______________________________________________________________________________
# TRY to import the cuda nearest neighbor pybind11 module -- if it exists in
# the path and we enabled `using_cuda`

from importlib.util import find_spec

KNN_LOADER    = find_spec("spinifel.sequential.pyCudaKNearestNeighbors")
KNN_AVAILABLE = KNN_LOADER is not None
if settings.verbose:
    print(f"pyCudaKNearestNeighbors is available: {KNN_AVAILABLE}")

if settings.using_cuda and KNN_AVAILABLE:
    import spinifel.sequential.pyCudaKNearestNeighbors as pyCu

#-------------------------------------------------------------------------------


def nearest_neighbor(model_slices, slices):
    if settings.using_cuda:
        if settings.verbose:
            print("Using CUDA nearest neighbor.")

        model_slices_flat = model_slices.flatten()
        slices_flat       = slices.flatten()
        
        deviceId = rank % settings._devices_per_node
        print(f"Rank {rank} using deviceId {deviceId}")

        euDist = pyCu.cudaEuclideanDistance(slices_flat,
                                            model_slices_flat,
                                            slices.shape[0],
                                            model_slices.shape[0],
                                            slices.shape[1],
                                            deviceId)
        index =  pyCu.cudaHeapSort(euDist,
                                   slices.shape[0],
                                   model_slices.shape[0],
                                   slices.shape[1],
                                   1,
                                   deviceId)
    else:
        if settings.verbose:
            print("Using sklearn nearest neighbor.")

        ed    = euclidean_distances(model_slices, slices)
        index = np.argmin(ed, axis=0)

    return index

