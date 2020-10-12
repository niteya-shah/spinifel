import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import os
import spinifel.sequential.pyCudaKNearestNeighbors as pyCu

def nearest_neighbor(model_slices, slices):
    if os.environ.get("USING_CUDA") == "1":
        print("Using CUDA nearest neighbor.")
        model_slices_flat = model_slices.flatten()
        slices_flat = slices.flatten()
        euDist = pyCu.cudaEuclideanDistance(slices_flat,
                                            model_slices_flat,
                                            slices.shape[0],
                                            model_slices.shape[0],
                                            slices.shape[1])
        index = pyCu.cudaHeapSort(euDist,
                                  slices.shape[0],
                                  model_slices.shape[0],
                                  slices.shape[1],
                                  1)
    else:
        print("Using sklearn nearest neighbor.")
        ed = euclidean_distances(model_slices, slices)
        index = np.argmin(ed, axis=0)
        
    return index

