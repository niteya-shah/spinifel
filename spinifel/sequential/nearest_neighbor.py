import os
import numpy as np
from   sklearn.metrics.pairwise import euclidean_distances

import spinifel.sequential.pyCudaKNearestNeighbors as pyCu
from   spinifel import SpinifelSettings


settings = SpinifelSettings()

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

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

