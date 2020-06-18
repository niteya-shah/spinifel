import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import pysingfel as ps

from spinifel import parms, utils
from spinifel.sequential.autocorrelation import forward
# Want sequential version even with MPI


def match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal):
    Mquat = parms.Mquat
    M = 4 * Mquat + 1
    N_orientations = 1000
    N_pixels = utils.prod(parms.reduced_det_shape)
    N_slices = slices_.shape[0]
    assert slices_.shape == (N_slices,) + parms.reduced_det_shape
    N = N_pixels * N_orientations
    ref_orientations = ps.get_uniform_quat(N_orientations, True)
    ref_rotmat = np.array([ps.quaternion2rot3d(quat) for quat in ref_orientations])
    H, K, L = np.einsum("ijk,klmn->jilmn", ref_rotmat, pixel_position_reciprocal)
    real_extent = 2
    reciprocal_extent = pixel_distance_reciprocal.max()
    H_ = H.flatten() / reciprocal_extent * np.pi
    K_ = K.flatten() / reciprocal_extent * np.pi
    L_ = L.flatten() / reciprocal_extent * np.pi
    model_slices = forward(ac, H_, K_, L_, 1, M, N, reciprocal_extent, True).real
    # Imaginary part ~ numerical error
    model_slices = model_slices.reshape((N_orientations, N_pixels))
    slices_ = slices_.reshape((N_slices, N_pixels))
    data_model_scaling_ratio = slices_.std() / model_slices.std()
    print(f"Data/Model std ratio: {data_model_scaling_ratio}.")
    model_slices *= data_model_scaling_ratio

    ed = euclidean_distances(model_slices, slices_)
    ed_argmin = np.argmin(ed, axis=0)

    return ref_orientations[ed_argmin]