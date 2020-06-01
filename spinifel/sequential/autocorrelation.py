import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import pysingfel as ps

from spinifel import parms


def solve_ac(N_images,
             pixel_position_reciprocal,
             pixel_distance_reciprocal,
             slices_):
    orientations = ps.get_random_quat(N_images)
    rotmat = np.array([ps.quaternion2rot3d(quat) for quat in orientations])
    H, K, L = np.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape

    Mquat = 10
    M = 4 * Mquat + 1
    real_extent = 2
    reciprocal_extent = pixel_distance_reciprocal.max()

    idx = np.abs(L) < reciprocal_extent * .01
    plt.scatter(H[idx], K[idx], c=slices_[idx], s=1, norm=LogNorm())
    plt.axis('equal')
    plt.colorbar()
    plt.savefig(parms.out_dir / "star_0.png")
    plt.cla()
    plt.clf()
