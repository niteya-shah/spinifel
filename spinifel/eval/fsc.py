import matplotlib.pyplot as plt
import argparse
import time
import os
import numpy as np
import h5py
import mrcfile
import scipy.interpolate
from align import save_mrc, align_volumes
import matplotlib
matplotlib.use('Agg')

"""
Estimate resolution based on the Fourier shell correlation (FSC) between 
the output MRC file and a reference density map generated from a PDB file.
"""

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(
        description="Simulate a simple SPI dataset.")
    parser.add_argument(
        '-m', '--mrc_file', help='mrc file of reconstructed map', required=True, type=str)
    parser.add_argument(
        '-d', '--dataset', help='h5 file of simulated data', required=True, type=str)
    parser.add_argument(
        '-p', '--pdb_file', help='pdb file for reference structure', required=True, type=str)
    parser.add_argument(
        '-o', '--output', help='output directory for aligned volumes', required=False, type=str)
    # optional arguments to adjust alignment protocol
    parser.add_argument('--zoom', help='Zoom factor during alignment',
                        required=False, type=float, default=1)
    parser.add_argument('--sigma', help='Sigma for Gaussian filtering during alignment',
                        required=False, type=float, default=0)
    parser.add_argument('--niter', help='Number of alignment iterations to run',
                        required=False, type=int, default=10)
    parser.add_argument('--nsearch', help='Number of quaternions to score per iteration',
                        required=False, type=int, default=360)
    parser.add_argument('--use-cupy', help='Use CuPy for GPU accelerated FSC calculation',
                        required=False, type=bool, default=True)

    return vars(parser.parse_args())

def get_reciprocal_mesh(voxel_number_1d, distance_reciprocal_max, xp):
    """
    Get a centered, symetric mesh of given dimensions. Altered from skopi.

    Parameters
    ----------
    voxel_number_1d : int
        number of voxels per axis
    distance_reciprocal_max : float
        maximum voxel resolution in inverse Angstrom

    Returns
    -------
    reciprocal_mesh : numpy.ndarray, shape (n,n,n,3)
        grid of reciprocal space vectors for each voxel
    """
    max_value = distance_reciprocal_max
    linspace = xp.linspace(-max_value, max_value, voxel_number_1d)
    reciprocal_mesh_stack = xp.asarray(
        xp.meshgrid(linspace, linspace, linspace, indexing='ij'))
    reciprocal_mesh = xp.moveaxis(reciprocal_mesh_stack, 0, -1)

    return reciprocal_mesh

def compute_reference(pdb_file, M, distance_reciprocal_max, xp):
    """
    Compute the reference density map from a PDB file using skopi.

    Parameters
    ----------
    pdb_file : string
        path to coordinates file in pdb format
    M : int
        number of voxels along each dimension of map
    distance_reciprocal_max : floa
        maximum voxel resolution in inverse Angstrom

    Returns
    -------
    density : numpy.ndarray, shape (M,M,M)
        reference density map
    """
    import skopi.gpu as pg
    import skopi as sk

    # set up Particle object
    particle = sk.Particle()
    particle.read_pdb(pdb_file, ff='WK')

    # compute ideal diffraction volume and take FT for density map
    mesh = get_reciprocal_mesh(M, distance_reciprocal_max, np)
    cfield = pg.calculate_diffraction_pattern_gpu(
        mesh, particle, return_type='complex_field')
    ivol = np.square(np.abs(cfield))
    density = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(cfield))).real

    return density

def compute_fsc(volume1, volume2, distance_reciprocal_max, xp, spacing=0.01, output=None):
    """
    Compute the Fourier shell correlation (FSC) curve, with the 
    estimated resolution based on a threshold of 0.5.

    Parameters
    ----------
    volume1 : numpy.ndarray, shape (n,n,n)
        reference map
    volume2 : numpy.ndarray, shape (n,n,n)
        reconstructed map
    distance_reciprocal_max : float
        maximum voxel resolution in inverse Angstrom
    spacing : float
        spacing for evaluating FSC in inverse Angstrom
    output : string, optional
        directory to which to save png of FSC curve

    Returns
    -------
    resolution : float
        estimated resolution of reconstructed map in Angstroms
    """

    mesh = get_reciprocal_mesh(volume1.shape[0], distance_reciprocal_max, xp)
    smags = xp.linalg.norm(xp.array(mesh), axis=-1).reshape(-1) * 1e-10
    volume1 = xp.array(volume1)
    volume2 = xp.array(volume2)
    r_spacings = xp.arange(0, smags.max() / np.sqrt(3), spacing)

    ft1 = xp.fft.fftshift(xp.fft.fftn(volume1)).reshape(-1)
    ft2 = xp.conjugate(xp.fft.fftshift(xp.fft.fftn(volume2)).reshape(-1))
    rshell, fsc = xp.zeros(len(r_spacings)), xp.zeros(len(r_spacings))

    for i, r in enumerate(r_spacings):
        indices = xp.where((smags > r) & (smags < r + spacing))[0]
        numerator = xp.sum(ft1[indices] * ft2[indices])
        denominator = xp.sqrt(xp.sum(
            xp.square(xp.abs(ft1[indices]))) * xp.sum(xp.square(xp.abs(ft2[indices]))))
        rshell[i] = r + 0.5 * spacing
        fsc[i] = numerator.real / denominator

    if not isinstance(fsc, np.ndarray):
        fsc = fsc.get()
        rshell = rshell.get()

    f = scipy.interpolate.interp1d(fsc, rshell)
    try:
        resolution = 1.0 / f(0.5)
        print(f"Estimated resolution from FSC: {resolution:.1f} Angstrom")
    except ValueError:
        resolution = -1
        print("Resolution could not be estimated.")

    # optionally plot
    if output is not None:
        f, ax1 = plt.subplots(figsize=(5,3))
        ax1.plot(rshell,fsc, c='black')
        ax1.scatter(rshell,fsc, c='black')
        ax1.plot([rshell.min(),rshell.max()],[0.5,0.5], c='grey', linestyle='dashed')
        ax1.set_xlim(rshell.min(),rshell.max())
        ax1.set_xlabel("Resolution (1/${\mathrm{\AA}}$)")
        ax1.set_ylabel("FSC", fontsize=12)
        f.savefig(os.path.join(output, "fsc.png"), dpi=300, bbox_inches='tight')

    return resolution

def main():

    args = parse_input()
    if not os.path.isdir(args['output']):
        os.mkdir(args['output'])
    if args['use_cupy']:
        import cupy as xp
        from cupyx.scipy import ndimage
    else:
        xp = np
        from scipy import ndimage

    # load and prepare input files
    volume = mrcfile.open(args['mrc_file']).data.copy()
    with h5py.File(args['dataset'], "r") as f:
        dist_recip_max = np.linalg.norm(
            f['pixel_position_reciprocal'][:], axis=-1).max()
    reference = compute_reference(
        args['pdb_file'], volume.shape[0], dist_recip_max, xp)

    # align volumes
    ali_volume, ali_reference = align_volumes(volume, reference, xp, ndimage, zoom=args['zoom'], sigma=args['sigma'],
                                              n_iterations=args['niter'], n_search=args['nsearch'])
    if args['output'] is not None:
        save_mrc(os.path.join(args['output'], "reference.mrc"), ali_reference)
        save_mrc(os.path.join(args['output'], "aligned.mrc"), ali_volume)

    # compute fsc
    resolution = compute_fsc(ali_reference, ali_volume,
                             dist_recip_max, xp, output=args['output'])

if __name__ == '__main__':
    main()
