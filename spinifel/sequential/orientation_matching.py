import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import pysingfel as ps

from spinifel import parms, utils, autocorrelation

#Packages for CUDA 
import os
import spinifel.sequential.pyCudaKNearestNeighbors as pyCu

def match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal):
    Mquat = parms.Mquat
    M = 4 * Mquat + 1
    N_orientations = parms.N_orientations
    N_pixels = utils.prod(parms.reduced_det_shape)
    N_slices = slices_.shape[0]
    assert slices_.shape == (N_slices,) + parms.reduced_det_shape
    N = N_pixels * N_orientations

    if not N_slices:
        return np.zeros((0, 4))

    ref_orientations = ps.get_uniform_quat(N_orientations, True)
    ref_rotmat = np.array([ps.quaternion2rot3d(quat) for quat in ref_orientations])
    H, K, L = np.einsum("ijk,klmn->jilmn", ref_rotmat, pixel_position_reciprocal)
    real_extent = 2
    reciprocal_extent = pixel_distance_reciprocal.max()
    H_ = H.flatten() / reciprocal_extent * np.pi / parms.oversampling
    K_ = K.flatten() / reciprocal_extent * np.pi / parms.oversampling
    L_ = L.flatten() / reciprocal_extent * np.pi / parms.oversampling
    model_slices = autocorrelation.forward(
        ac, H_, K_, L_, 1, M, N, reciprocal_extent, True).real
    # Imaginary part ~ numerical error
    model_slices = model_slices.reshape((N_orientations, N_pixels))
    slices_ = slices_.reshape((N_slices, N_pixels))
    data_model_scaling_ratio = slices_.std() / model_slices.std()
    print(f"Data/Model std ratio: {data_model_scaling_ratio}.", flush=True)
    model_slices *= data_model_scaling_ratio

    if os.environ.get("USING_CUDA") == "1":
        print("Implementing nearest neighbor using CUDA.")
        model_slices_flat = model_slices.flatten()
        slices_flat = slices_.flatten()
        euDist = pyCu.cudaEuclideanDistance(slices_flat,model_slices_flat,slices_.shape[0],model_slices.shape[0],slices_.shape[1])
        index = pyCu.cudaHeapSort(euDist,slices_.shape[0],model_slices.shape[0],slices_.shape[1],1)
#        index = np.zeros(slices_.shape[0],dtype=np.int32)
#        print("Shape of model_slices_flat: ",model_slices_flat.shape,", slices_flat: ",slices_flat.shape," and index: ",index.shape)
#        print("index values: ",index[:])
    else:
        print("Implementing nearest neighbor using sklearn package.")
        ed = euclidean_distances(model_slices, slices_)
        index = np.argmin(ed, axis=0)
#        print("Shape of model_slices: ",model_slices.shape,", slices_: ",slices_.shape," and index: ",index.shape)
#        print("index values: ",index[:])

    return ref_orientations[index]
