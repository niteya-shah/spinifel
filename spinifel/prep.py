import h5py
import sys
import matplotlib.pyplot as plt
import numpy             as np
import mrcfile
import PyNVTX            as nvtx

from matplotlib.colors import LogNorm

from spinifel import settings


@nvtx.annotate("prep.py", is_prefix=True)
def get_saxs(pixel_distance_reciprocal, mean_image, mask=1,
             filter=False, filter_order=2, filter_threshold=0.25,
            threshold=10):
    """
    Compute the radial intensity profile of an image.
    
    Parameters
    ----------
    pixel_distance_reciprocal: numpy.ndarray
    mean_image : numpy.ndarray, shape (n,m)
        detector image
    mask : numpy.ndarray, shape (n,m)
        detector mask, with zeros corresponding to pixels to mask
    filter : bool
        if True, apply a lowpass Butterworth filter to the radial intensity profile
    filter_order : int
        order of the Butterworth filter
    filter_threshold : float
        critical frequency for the bandpass Butterworth filter
    threshold : float
        below this intensity, set the radial intensity profile to zero
        
    Returns
    -------
    np.linspace(0, q_max, N+1): numpy.ndarray, 1d
        q range
    saxs : numpy.ndarray, 1d
        radial intensity profile of input image
    """
    qs = pixel_distance_reciprocal[mask!=0].flatten()
    N = 1000
    q_max = qs.max()
    idx = (N * qs / q_max).astype(np.int)
    saxs_acc = np.bincount(idx, mean_image.flatten(), N)
    saxs_wgt = np.bincount(idx, None, N)
    saxs = saxs_acc / saxs_wgt
    return np.linspace(0, q_max, N + 1), saxs



@nvtx.annotate("prep.py", is_prefix=True)
def export_saxs(pixel_distance_reciprocal, mean_image, filename):
    saxs_qs, saxs = get_saxs(pixel_distance_reciprocal, mean_image)
    plt.semilogy(saxs_qs, saxs)
    plt.savefig(settings.out_dir / filename)
    plt.cla()
    plt.clf()



@nvtx.annotate("prep.py", is_prefix=True)
def bin2x2_sum(arr):
    return (arr[..., ::2, ::2] + arr[..., 1::2, ::2] + arr[..., ::2, 1::2]
            + arr[..., 1::2, 1::2])



@nvtx.annotate("prep.py", is_prefix=True)
def bin2x2_mean(arr):
    return bin2x2_sum(arr) / 4



@nvtx.annotate("prep.py", is_prefix=True)
def bin2x2_index(arr):
    arr = np.minimum(arr[..., ::2, :], arr[..., 1::2, :])
    arr = np.minimum(arr[..., ::2], arr[..., 1::2])
    return arr // 2



@nvtx.annotate("prep.py", is_prefix=True)
def bin2nx2n_sum(arr, n):
    for _ in range(n):
        arr = bin2x2_sum(arr)
    return arr



@nvtx.annotate("prep.py", is_prefix=True)
def bin2nx2n_mean(arr, n):
    for _ in range(n):
        arr = bin2x2_mean(arr)
    return arr



@nvtx.annotate("prep.py", is_prefix=True)
def bin2nx2n_index(arr, n):
    for _ in range(n):
        arr = bin2x2_index(arr)
    return arr



@nvtx.annotate("prep.py", is_prefix=True)
def clipping(arr, n):
    """
    Clip/truncate high resolution region from input. Currently only valid for
    PnCCD or monolithic detectors.
    :param arr: array to clip
    :param n: clipping factor, with image size reduced by a factor of 2**n
    :return narr: clipped array
    """
    n = 2**n
    sa, sb = arr.shape[-2:]
    narr = np.zeros(arr.shape[:-2] + (sa // n, sb // n), dtype=arr.dtype)
    if narr.shape[1] == 4: # valid for earlier PnCCD, possibly not current
        narr[..., 0, :, :] = arr[..., 0, -sa // n:, -sb // n:]
        narr[..., 1, :, :] = arr[..., 1, -sa // n:, :sb // n]
        narr[..., 2, :, :] = arr[..., 2, -sa // n:, -sb // n:]
        narr[..., 3, :, :] = arr[..., 3, -sa // n:, :sb // n]
    elif narr.shape[1] == 1: # valid for monolithic
        n *= 2
        #narr[..., 0, :, :] = arr[..., n_panel,sa//2-sa//n:sa//2+sa//n,sb//2-sb//n:sb//2+sb//n]
        narr[..., 0, :, :] = arr[..., 0, sa // 2 - sa // n:sa //
                                 2 + sa // n, sb // 2 - sb // n:sb // 2 + sb // n]
    else:
        print(
            "Clipping function doesn't currently accept a detector with %i panels" %
            narr.shape[1])
    return narr



@nvtx.annotate("prep.py", is_prefix=True)
def clipping_index(arr, n):
    arr = clipping(arr, n)
    #for i in range(2):
    #    arr[i] -= arr[i].min()
    arr -= arr.min()
    return arr


def binning_sum(arr): return bin2nx2n_sum(
    clipping(arr, settings.N_clipping), settings.N_binning)
def binning_mean(arr): return bin2nx2n_mean(
    clipping(arr, settings.N_clipping), settings.N_binning)
def binning_index(arr): return bin2nx2n_index(
    clipping_index(arr, settings.N_clipping), settings.N_binning)



@nvtx.annotate("prep.py", is_prefix=True)
def load_pixel_position_reciprocal(pixel_position_reciprocal):
    with h5py.File(settings.data_path, 'r') as h5f:
        pixel_position_reciprocal[:] = np.moveaxis(
            h5f['pixel_position_reciprocal'][:], -1, 0)



@nvtx.annotate("prep.py", is_prefix=True)
def load_pixel_index_map(pixel_index_map):
    with h5py.File(settings.data_path, 'r') as h5f:
        pixel_index_map[:] = np.moveaxis(
            h5f['pixel_index_map'][:], -1, 0)



@nvtx.annotate("prep.py", is_prefix=True)
def load_slices(slices, i_start, i_end):
    """Populate intensity slices from input file."""
    with h5py.File(settings.data_path, 'r') as h5f:
        if h5f['intensities'].shape[0] >= i_end:
            slices[:] = h5f['intensities'][i_start:i_end]
        else:
            sys.exit(
                f"Error: Not enough intensity slices (max:{h5f['intensities'].shape[0]:d})")

@nvtx.annotate("prep.py", is_prefix=True)
def load_orientations(orientations, i_start, i_end):
    with h5py.File(settings.data_path, 'r') as h5f:
        orientations[:] = h5f['orientations'][i_start:i_end]



@nvtx.annotate("prep.py", is_prefix=True)
def load_volume(volume):
    with h5py.File(settings.data_path, 'r') as h5f:
        volume[:] = h5f['volume']



@nvtx.annotate("prep.py", is_prefix=True)
def compute_pixel_distance(pixel_position_reciprocal):
    """pixel_position_reciprocal is of shape [q, # of panels, # of rows, # of columns], 
    where the first two dimensions of q corresponds to qx and qy."""
    return np.sqrt((pixel_position_reciprocal[:2,:,:,:]**2).sum(axis=0))
    #return np.sqrt((pixel_position_reciprocal[:2]**2).sum(axis=0))



@nvtx.annotate("prep.py", is_prefix=True)
def load_orientations_prior(orientations_prior, i_start, i_end):
    with h5py.File(settings.data_path, 'r') as h5f:
        orientations = h5f['orientations'][i_start:i_end]
        orientations_prior[:] = np.reshape(
            orientations, (orientations.shape[0], 4))



@nvtx.annotate("prep.py", is_prefix=True)
def save_mrc(savename, data, voxel_size=None):
    """
    Save Nd numpy array to path savename in mrc format.

    :param savename: path to which to save mrc file
    :param data: input numpy array
    :param voxel_size: voxel size for header, optional
    """

    mrc = mrcfile.new(savename, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(data.astype(np.float32))
    if voxel_size is not None:
        mrc.voxel_size = voxel_size
    mrc.close()
    return


@nvtx.annotate("prep.py", is_prefix=True)
def load_pixel_position_reciprocal_psana(
        run, pixel_position, pixel_position_reciprocal):
    if run.expt == "xpptut15":
        pixel_position_reciprocal[:] = np.moveaxis(
            run.beginruns[0].scan[0].raw.pixel_position_reciprocal[:], -1, 0)
    elif run.expt == "amo06516":
        pixel_position = run.beginruns[0].scan[0].pixel_position
    else:
        assert False
