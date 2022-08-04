import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from eval import config
# Monkey patch for numpy
config.xp = np
config.ndimage = ndimage
from eval.fsc import compute_fsc, compute_reference
from eval.align import rotate_volume
import skopi as skp
import os
import mrcfile

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
        self.ref_density = compute_reference(args['pdb_file'], args['M'], 1e10/args['resolution'])
        
        self.args = args

    def test_rotate_volume(self):
        """ Check that volume rotation works """

        q_tar = skp.get_random_quat(1)[0]
        tar_density = rotate_volume(self.ref_density, np.linalg.inv(skp.quaternion2rot3d(q_tar)))
        inv_density = rotate_volume(tar_density, skp.quaternion2rot3d(q_tar))
        assert np.corrcoef(self.ref_density.flatten(), inv_density.flatten())[0,1] > 0.85

        f, ((ax1,ax2,ax3)) = plt.subplots(1, 3, figsize=(9,3))
        hs = int(self.args['M']/2)

        ax1.imshow(self.ref_density[hs,:,:])
        ax2.imshow(tar_density[hs,:,:])
        ax3.imshow(inv_density[hs,:,:])
        
        ax1.set_title("Reference", fontsize=12)
        ax2.set_title("Rotated", fontsize=12)
        ax3.set_title("Unrotated", fontsize=12)

        f.savefig("test_rotate_volume.png", dpi=300, bbox_inches='tight')

    def test_compute_fsc(self):
        """ Check volume aligntment and FSC """
        q_tar = skp.get_random_quat(1)[0]
        tar_density = rotate_volume(self.ref_density, np.linalg.inv(skp.quaternion2rot3d(q_tar)))
        inv_density = rotate_volume(tar_density, skp.quaternion2rot3d(q_tar))
        
        rs, fsc, res = compute_fsc(self.ref_density, inv_density, 1e10/self.args['resolution'], self.args['spacing'])
        assert res < self.args['resolution']*1.25