import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from spinifel import parms


def get_saxs(pixel_distance_reciprocal, mean_image):
    qs = pixel_distance_reciprocal.flatten()
    N = 100
    q_max = qs.max()
    idx = (N*qs/q_max).astype(np.int)
    saxs_acc = np.bincount(idx, mean_image.flatten(), N)
    saxs_wgt = np.bincount(idx, None, N)
    saxs = saxs_acc / saxs_wgt
    return np.linspace(0, q_max, N+1), saxs


def export_saxs(pixel_distance_reciprocal, mean_image, filename):
    saxs_qs, saxs = get_saxs(pixel_distance_reciprocal, mean_image)
    plt.semilogy(saxs_qs, saxs)
    plt.savefig(parms.out_dir / filename)
    plt.cla()
    plt.clf()


def bin2x2_sum(arr):
    return (arr[..., ::2, ::2] + arr[..., 1::2, ::2] + arr[..., ::2, 1::2]
            + arr[..., 1::2, 1::2])


def bin2x2_mean(arr):
    return bin2x2_sum(arr) / 4


def bin2x2_index(arr):
    arr = np.minimum(arr[..., ::2, :], arr[..., 1::2, :])
    arr = np.minimum(arr[..., ::2], arr[..., 1::2])
    return arr // 2


def bin2nx2n_sum(arr, n):
    for _ in range(n):
        arr = bin2x2_sum(arr)
    return arr


def bin2nx2n_mean(arr, n):
    for _ in range(n):
        arr = bin2x2_mean(arr)
    return arr


def bin2nx2n_index(arr, n):
    for _ in range(n):
        arr = bin2x2_index(arr)
    return arr


def clipping(arr, n):
    n = 2**n
    sa, sb = arr.shape[-2:]
    narr = np.zeros(arr.shape[:-2] + (sa//n, sb//n), dtype=arr.dtype)
    narr[..., 0, :, :] = arr[..., 0, -sa//n:, -sb//n:]
    narr[..., 1, :, :] = arr[..., 1, -sa//n:, :sb//n]
    narr[..., 2, :, :] = arr[..., 2, -sa//n:, -sb//n:]
    narr[..., 3, :, :] = arr[..., 3, -sa//n:, :sb//n]
    return narr


def clipping_index(arr, n):
    arr = clipping(arr, n)
    for i in range(2):
        arr[i] -= arr[i].min()
    return arr


binning_sum = lambda arr: bin2nx2n_sum(
    clipping(arr, parms.N_clipping), parms.N_binning)
binning_mean = lambda arr: bin2nx2n_mean(
    clipping(arr, parms.N_clipping), parms.N_binning)
binning_index = lambda arr: bin2nx2n_index(
    clipping_index(arr, parms.N_clipping), parms.N_binning)


def load_pixel_position_reciprocal(pixel_position_reciprocal):
    with h5py.File(parms.data_path, 'r') as h5f:
        pixel_position_reciprocal[:] = np.moveaxis(
            h5f['pixel_position_reciprocal'][:], -1, 0)


def load_pixel_index_map(pixel_index_map):
    with h5py.File(parms.data_path, 'r') as h5f:
        pixel_index_map[:] = np.moveaxis(
            h5f['pixel_index_map'][:], -1, 0)


def load_slices(slices, i_start, i_end):
    with h5py.File(parms.data_path, 'r') as h5f:
        slices[:] = h5f['intensities'][i_start:i_end]


def load_orientations(orientations, i_start, i_end):
    with h5py.File(parms.data_path, 'r') as h5f:
        orientations[:] = h5f['orientations'][i_start:i_end]

def load_volume(volume):
    with h5py.File(parms.data_path, 'r') as h5f:
        volume[:] = h5f['volume']

def compute_pixel_distance(pixel_position_reciprocal):
    return np.sqrt((pixel_position_reciprocal**2).sum(axis=0))
