import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpi4py import MPI

from spinifel import parms, prep


def get_data(N_images_per_rank):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    pixel_position_reciprocal = np.zeros((3,) + parms.det_shape, dtype=np.float)
    pixel_index_map = np.zeros((2,) + parms.det_shape, dtype=np.int)
    if rank == 0:
        with h5py.File(parms.data_path, 'r') as h5f:
            pixel_position_reciprocal[:] = np.moveaxis(
                h5f['pixel_position_reciprocal'][:], -1, 0)
            pixel_index_map[:] = np.moveaxis(
                h5f['pixel_index_map'][:], -1, 0)
    comm.Bcast(pixel_position_reciprocal, root=0)
    comm.Bcast(pixel_index_map, root=0)

    data_type = getattr(np, parms.data_type_str)
    slices_ = np.zeros((N_images_per_rank,) + parms.det_shape, dtype=data_type)
    i_start = rank * N_images_per_rank
    i_end = i_start + N_images_per_rank
    with h5py.File(parms.data_path, 'r') as h5f:
        slices_[:] = h5f['intensities'][i_start:i_end]

    if rank == 0:
        prep.show_image(pixel_index_map, slices_[0], "image_0.png")

    mean_image = slices_.mean(axis=0)
    reduced_image = np.zeros_like(mean_image)
    comm.Reduce(mean_image, reduced_image, op=MPI.SUM, root=0)
    if rank == 0:
        mean_image_tot = reduced_image / size
        prep.show_image(pixel_index_map, mean_image_tot, "mean_image.png")
        pixel_distance_reciprocal = np.sqrt(
            (pixel_position_reciprocal**2).sum(axis=0))
        saxs_qs, saxs = prep.get_saxs(pixel_distance_reciprocal, mean_image_tot)
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

    if rank == 0:
        pixel_position_reciprocal = binning_mean(pixel_position_reciprocal)
        pixel_index_map = binning_index(pixel_index_map)
    slices_ = binning_sum(slices_)

    if rank == 0:
        prep.show_image(pixel_index_map, slices_[0], "image_binned_0.png")

    mean_image = slices_.mean(axis=0)
    reduced_image = np.zeros_like(mean_image)
    comm.Reduce(mean_image, reduced_image, op=MPI.SUM, root=0)
    if rank == 0:
        mean_image_tot = reduced_image / size
        prep.show_image(pixel_index_map, mean_image_tot, "mean_image_binned.png")

    pixel_distance_reciprocal = np.sqrt(
        (pixel_position_reciprocal**2).sum(axis=0))
    if rank == 0:
        saxs_qs, saxs = prep.get_saxs(pixel_distance_reciprocal, mean_image_tot)
        plt.semilogy(saxs_qs, saxs)
        plt.savefig(parms.out_dir / "saxs_binned.png")
        plt.cla()
        plt.clf()

    return (pixel_position_reciprocal,
            pixel_distance_reciprocal,
            pixel_index_map,
            slices_)
