import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpi4py import MPI

from spinifel import parms, prep, image


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


def get_slices(comm, N_images_per_rank, ds):
    data_type = getattr(np, parms.data_type_str)
    slices_ = np.zeros((N_images_per_rank,) + parms.det_shape,
                       dtype=data_type)
    if ds is None:
        i_start = comm.rank * N_images_per_rank
        i_end = i_start + N_images_per_rank
        prep.load_slices(slices_, i_start, i_end)
        return slices_
    else:
        i = 0
        for run in ds.runs():
            for evt in run.events():
                raw = evt._dgrams[0].pnccdBack[0].raw
                slices_[i] = raw.image
                i += 1
                if i >= N_images_per_rank:
                    # Each big data node needs to receive a StopIteration
                    # from the Event Builder before the Event Builder
                    # can stop. However, we can't be guaranteed they'd
                    # ask the last chunk at the same time.
                    # So, we have a barrier until all are full and then
                    # ask for the StopIteration. If we don't get a
                    # StopIteration, `slices_[i]` will crash a few lines
                    # above.
                    bd_comm = comm.Create_group(comm.group.Excl([0, 1]))
                    bd_comm.Barrier()
        return slices_[:i]


def compute_mean_image(comm, slices_):
    images_sum = slices_.sum(axis=0)
    N_images = slices_.shape[0]
    images_sum_total = np.zeros_like(images_sum)
    comm.Reduce(images_sum, images_sum_total, op=MPI.SUM, root=0)
    N_images = comm.reduce(N_images, op=MPI.SUM, root=0)
    if comm.rank == 0:
        return images_sum_total / N_images
    else:
        # Send None rather than intermediary result
        return None


def get_data(N_images_per_rank, ds):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    pixel_position_reciprocal = get_pixel_position_reciprocal(comm)
    pixel_index_map = get_pixel_index_map(comm)

    slices_ = get_slices(comm, N_images_per_rank, ds)
    N_images_local = slices_.shape[0]
    witness = slices_.flatten()[0] if N_images_local else None
    print(f"Rank {comm.rank}: {N_images_local} values, start: {witness}", flush=True)
    mean_image = compute_mean_image(comm, slices_)

    if rank == (2 if parms.use_psana else 0):
        image.show_image(pixel_index_map, slices_[0], "image_0.png")

    if rank == 0:
        pixel_distance_reciprocal = prep.compute_pixel_distance(
            pixel_position_reciprocal)
        image.show_image(pixel_index_map, mean_image, "mean_image.png")
        prep.export_saxs(pixel_distance_reciprocal, mean_image, "saxs.png")

    pixel_position_reciprocal = prep.binning_mean(pixel_position_reciprocal)
    pixel_index_map = prep.binning_index(pixel_index_map)
    pixel_distance_reciprocal = prep.compute_pixel_distance(
            pixel_position_reciprocal)

    slices_ = prep.binning_sum(slices_)
    mean_image = compute_mean_image(comm, slices_)

    if rank == (2 if parms.use_psana else 0):
        image.show_image(pixel_index_map, slices_[0], "image_binned_0.png")

    if rank == 0:
        image.show_image(pixel_index_map, mean_image, "mean_image_binned.png")
        prep.export_saxs(pixel_distance_reciprocal, mean_image,
                         "saxs_binned.png")

    return (pixel_position_reciprocal,
            pixel_distance_reciprocal,
            pixel_index_map,
            slices_)
