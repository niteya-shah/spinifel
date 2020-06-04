import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpi4py import MPI

from spinifel import parms, prep


def get_pixel_position_reciprocal(comm):
    pixel_position_type = getattr(np, parms.pixel_position_type_str)
    pixel_position_reciprocal = np.zeros(parms.pixel_position_shape,
                                         dtype=pixel_position_type)
    if comm.rank == 0:
        prep.load_pixel_position_reciprocal(pixel_position_reciprocal)
    comm.Bcast(pixel_position_reciprocal, root=0)
    return pixel_position_reciprocal


def get_pixel_index_map(comm):
    pixel_index_type = getattr(np, parms.pixel_index_type_str)
    pixel_index_map = np.zeros(parms.pixel_index_shape,
                               dtype=pixel_index_type)
    if comm.rank == 0:
        prep.load_pixel_index_map(pixel_index_map)
    comm.Bcast(pixel_index_map, root=0)
    return pixel_index_map


def get_slices(comm, N_images_per_rank):
    data_type = getattr(np, parms.data_type_str)
    slices_ = np.zeros((N_images_per_rank,) + parms.det_shape,
                       dtype=data_type)
    i_start = comm.rank * N_images_per_rank
    i_end = i_start + N_images_per_rank
    with h5py.File(parms.data_path, 'r') as h5f:
        slices_[:] = h5f['intensities'][i_start:i_end]
    return slices_


def get_data(N_images_per_rank):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    pixel_position_reciprocal = get_pixel_position_reciprocal(comm)
    pixel_index_map = get_pixel_index_map(comm)

    slices_ = get_slices(comm, N_images_per_rank)

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

    if rank == 0:
        pixel_position_reciprocal = prep.binning_mean(pixel_position_reciprocal)
        pixel_index_map = prep.binning_index(pixel_index_map)
    slices_ = prep.binning_sum(slices_)

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
