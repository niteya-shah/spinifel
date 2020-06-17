import numpy as np

import pysingfel as ps

from spinifel import parms, utils
from spinifel.sequential.autocorrelation import forward
# Want sequential version even with MPI


def match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal):
    Mquat = parms.Mquat
    M = 4 * Mquat + 1
    or_N = 1000
    N = utils.prod(parms.reduced_det_shape) * or_N
    ref_orientations = ps.get_uniform_quat(or_N, True)
    ref_rotmat = np.array([ps.quaternion2rot3d(quat) for quat in ref_orientations])
    H, K, L = np.einsum("ijk,klmn->jilmn", ref_rotmat, pixel_position_reciprocal)
    real_extent = 2
    reciprocal_extent = pixel_distance_reciprocal.max()
    H_ = H.flatten() / reciprocal_extent * np.pi
    K_ = K.flatten() / reciprocal_extent * np.pi
    L_ = L.flatten() / reciprocal_extent * np.pi
    model_slices = forward(ac, H_, K_, L_, 1, M, N, reciprocal_extent, True)
