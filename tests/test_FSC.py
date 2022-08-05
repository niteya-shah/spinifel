import mrcfile
import os
import skopi as skp
import numpy as np
from scipy import ndimage

from eval import config
# Monkey patch for numpy
config.xp = np
config.ndimage = ndimage
from eval.align import rotate_volume
from eval.fsc import compute_fsc, compute_reference




class TestFSC(object):
    """
    Test the FSC calculation by rotating a volume and then testing FSC to be within margin of error
    """

    @classmethod
    def setup_class(self):
        args = {}
        args['pdb_file'] = "/pscratch/sd/n/niteya56/run/3iyf.pdb"
        args['resolution'] = 9.0 # in Angstrom
        args['M'] = 81
        args['spacing'] = 0.01
        self.ref_density = compute_reference(
            args['pdb_file'], args['M'], 1e10 / args['resolution'])
        q_tar = skp.get_random_quat(1)
        R_inv = np.linalg.inv(skp.quaternion2rot3d(q_tar[0]))
        q_tar_inv = np.expand_dims(skp.rotmat_to_quaternion(R_inv), axis=0)
        tar_density = rotate_volume(self.ref_density, q_tar_inv)
        self.inv_density = rotate_volume(tar_density, q_tar)

        self.args = args

    def test_rotate_volume(self):
        """ Check that volume rotation works """
        assert np.corrcoef(
            self.ref_density.flatten(),
            self.inv_density.flatten())[0, 1] > 0.85

    def test_compute_fsc(self):
        """ Check volume aligntment and FSC """
        rs, fsc, res = compute_fsc(
            self.ref_density, self.inv_density, 1e10 / self.args['resolution'], self.args['spacing'])
        assert res < self.args['resolution'] * 1.25
