# Test skopi functions used in spinifel

import skopi as skp
import numpy as np


def test_quat():

    N_orientations = 1

    # This generate of list of quaternions
    # For testing, just get the first quaternion (index = 0)
    #quat = skp.get_uniform_quat(N_orientations, avoid_symmetric=True)[0]
    print(f'Convert quaternion to rotation matrix')
    quat = np.array([0, 0, 0, 1], dtype=np.float64)
    print(f'quat={quat}')
    rotmat = skp.quaternion2rot3d(quat)
    print(f'rotmat={rotmat} ')

    print()
    print(f'Convert rotation matrix to quaternion')
    rotmat = np.array([[-1., 0.,  0.],
            [ 0., -1.,  0.],
            [ 0.,  0.,  1.]], dtype=np.float64)
    print(f'rotmat={rotmat}')
    o_quat = skp.rotmat_to_quaternion(rotmat)
    print(f'quat={o_quat}')


    assert np.array_equal(quat, o_quat)

if __name__ == "__main__":
    test_quat()
