# Test skopi functions used in spinifel

import skopi as skp
import numpy as np
from scipy.spatial.transform import Rotation as R

from psana import DataSource
from psana.psexp.tools import get_excl_ranks


def test_quat():
    # https://www.andre-gaschler.com/rotationconverter/
    # quaternion format is [w, x, y, z] (skopi)

    print(f"Convert quaternion to rotation matrix")
    quat = np.array([0, 0, 0, 1], dtype=np.float64)
    print(f"quat={quat}")
    rotmat = skp.quaternion2rot3d(quat)
    print(f"rotmat={rotmat} ")
    print()

    print(f"Convert rotation matrix to quaternion")
    rotmat = np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    print(f"rotmat={rotmat}")
    o_quat = skp.rotmat_to_quaternion(rotmat)
    print(f"quat={o_quat}")
    print()

    assert np.array_equal(quat, o_quat)


def test_np_rotations():
    # Format of scipy quaternion: x, y, z, w
    print(f"Scipy quarternion - rotation matrix conversion")
    quat = [0, 0, 0, 1]
    r = R.from_quat(quat)
    print(f"quat={r.as_quat()}")
    rotmat = r.as_matrix()
    print(f"rotmat={rotmat}")
    print()


if __name__ == "__main__":
    test_quat()
    test_np_rotations()
