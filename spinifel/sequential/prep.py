import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from spinifel import parms, prep


def get_data(N_images):
    with h5py.File(parms.data_path, 'r') as h5f:
        pixel_position_reciprocal = h5f['pixel_position_reciprocal'][:]
        pixel_index_map = h5f['pixel_index_map'][:]
        slices_ = h5f['intensities'][:N_images]
    pixel_position_reciprocal = np.moveaxis(pixel_position_reciprocal, -1, 0)
    pixel_index_map = np.moveaxis(pixel_index_map, -1, 0)

    prep.show_image(pixel_index_map, slices_[0], "image_0.png")

    mean_image = slices_.mean(axis=0)
    prep.show_image(pixel_index_map, mean_image, "mean_image.png")

    pixel_distance_reciprocal = np.sqrt(
        (pixel_position_reciprocal**2).sum(axis=0))
    saxs_qs, saxs = prep.get_saxs(pixel_distance_reciprocal, mean_image)
    plt.semilogy(saxs_qs, saxs)
    plt.savefig(parms.out_dir / "saxs.png")
    plt.cla()
    plt.clf()

    binning_sum = lambda arr: prep.bin2nx2n_sum(
        prep.clipping(arr, parms.N_clipping), parms.N_binning)
    binning_mean = lambda arr: prep.bin2nx2n_mean(
        prep.clipping(arr, parms.N_clipping), parms.N_binning)
    binning_index = lambda arr: prep.bin2nx2n_index(
        prep.clipping_index(arr, parms.N_clipping), parms.N_binning)

    pixel_position_reciprocal = binning_mean(pixel_position_reciprocal)
    pixel_index_map = binning_index(pixel_index_map)
    slices_ = binning_sum(slices_)

    prep.show_image(pixel_index_map, slices_[0], "image_binned_0.png")

    mean_image = slices_.mean(axis=0)
    prep.show_image(pixel_index_map, mean_image, "mean_image_binned.png")

    pixel_distance_reciprocal = np.sqrt(
        (pixel_position_reciprocal**2).sum(axis=0))
    saxs_qs, saxs = prep.get_saxs(pixel_distance_reciprocal, mean_image)
    plt.semilogy(saxs_qs, saxs)
    plt.savefig(parms.out_dir / "saxs_binned.png")
    plt.cla()
    plt.clf()

    return (pixel_position_reciprocal,
            pixel_distance_reciprocal,
            pixel_index_map,
            slices_)
