import os
import h5py
import mrcfile
import numpy as np
import sys
import argparse
from distutils import util
from eval import config

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(
        description="Simulate a simple SPI dataset.")
    parser.add_argument(
        '-m',
        '--mrc_file',
        help='mrc file of reconstructed map',
        required=True,
        type=str)
    parser.add_argument(
        '-d',
        '--dataset',
        help='h5 file of simulated data',
        required=True,
        type=str)
    parser.add_argument(
        '-p',
        '--pdb_file',
        help='pdb file for reference structure',
        required=True,
        type=str)
    parser.add_argument(
        '-o',
        '--output',
        help='output directory for aligned volumes',
        required=False,
        type=str)
    # optional arguments to adjust alignment protocol
    parser.add_argument('--zoom', help='Zoom factor during alignment',
                        required=False, type=float, default=1)
    parser.add_argument(
        '--sigma',
        help='Sigma for Gaussian filtering during alignment',
        required=False,
        type=float,
        default=0)
    parser.add_argument(
        '--niter',
        help='Number of alignment iterations to run',
        required=False,
        type=int,
        default=10)
    parser.add_argument(
        '--nsearch',
        help='Number of quaternions to score per iteration',
        required=False,
        type=int,
        default=360)
    parser.add_argument(
        '--use-cupy',
        help='Use CuPy for GPU accelerated FSC calculation',
        required=False,
        type=util.strtobool,
        default=True)


    return vars(parser.parse_args())
if __name__ == '__main__':

    args = parse_input()
    if args['use_cupy']:
        import cupy as xp
        from cupyx.scipy import ndimage
    else:
        xp = np
        from scipy import ndimage

    # Monkey patch our imports to allow command line arguments to change
    # imports
    config.xp = xp
    config.ndimage = ndimage
    from eval.fsc import compute_fsc, compute_reference, plot
    from eval.align import align_volumes, save_mrc

    if args['output'] is not None and not os.path.isdir(args['output']):
        os.mkdir(args['output'])

    # load and prepare input files
    volume = mrcfile.open(args['mrc_file']).data.copy()
    with h5py.File(args['dataset'], "r") as f:
        dist_recip_max = np.linalg.norm(
            f['pixel_position_reciprocal'][:], axis=-1).max()
    reference = compute_reference(
        args['pdb_file'], volume.shape[0], dist_recip_max)

    # align volumes
    ali_volume, ali_reference = align_volumes(volume, reference, zoom=args['zoom'], sigma=args['sigma'],
                                              n_iterations=args['niter'], n_search=args['nsearch'])
    if args['output'] is not None:
        save_mrc(os.path.join(args['output'], "reference.mrc"), ali_reference)
        save_mrc(os.path.join(args['output'], "aligned.mrc"), ali_volume)

    # compute fsc
    resolution, rshell, fsc_val = compute_fsc(
        ali_reference, ali_volume, dist_recip_max)

    # optionally plot
    if args['output'] is not None:
        plot(rshell, fsc_val, args['output'])
