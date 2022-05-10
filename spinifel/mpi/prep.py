import h5py
import matplotlib.pyplot as plt
import numpy as np
import PyNVTX as nvtx
from matplotlib.colors import LogNorm
from mpi4py import MPI

from spinifel import settings, prep, image, contexts


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_pixel_position_reciprocal(comm):
    """Rank0 broadcast pixel reciprocal positions from input file."""
    pixel_position_type = getattr(np, settings.pixel_position_type_str)
    pixel_position_reciprocal = np.zeros(settings.pixel_position_shape,
                                         dtype=pixel_position_type)
    if comm.rank == 0:
        prep.load_pixel_position_reciprocal(pixel_position_reciprocal)
    comm.Bcast(pixel_position_reciprocal, root=0)
    return pixel_position_reciprocal


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_pixel_index_map(comm):
    """Rank0 broadcast pixel index map from input file."""
    pixel_index_type = getattr(np, settings.pixel_index_type_str)
    pixel_index_map = np.zeros(settings.pixel_index_shape,
                               dtype=pixel_index_type)
    if comm.rank == 0:
        prep.load_pixel_index_map(pixel_index_map)
    comm.Bcast(pixel_index_map, root=0)
    return pixel_index_map


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_orientations_prior(comm, N_images_per_rank):
    """Each rank loads correct orientations for data images from input file."""
    data_type = getattr(np, settings.data_type_str)
    orientations_prior = np.zeros((N_images_per_rank,4),
                       dtype=data_type)
    i_start = comm.rank * N_images_per_rank
    i_end = i_start + N_images_per_rank
    print(f"get_orientations_prior rank={comm.rank} st={i_start} en={i_end}")
    prep.load_orientations_prior(orientations_prior, i_start, i_end)
    return orientations_prior


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_slices(comm, N_images_per_rank, ds):
    """Each rank loads intensity slices from input file (or psana)."""
    data_type = getattr(np, settings.data_type_str)
    slices_ = np.zeros((N_images_per_rank,) + settings.det_shape,
                       dtype=data_type)
    if ds is None:
        i_start = comm.rank * N_images_per_rank
        i_end = i_start + N_images_per_rank
        print(f"get_slices rank={comm.rank} st={i_start} en={i_end}")
        prep.load_slices(slices_, i_start, i_end)
        return slices_
    else:
        i = 0
        for run in ds.runs():
            for evt in run.events():
                raw = evt._dgrams[0].pnccdBack[0].raw
                try:
                    slices_[i] = raw.image
                except IndexError:
                    raise RuntimeError(
                        f"Rank {comm.rank} received too many events.")
                i += 1
        return slices_[:i]



@nvtx.annotate("mpi/prep.py", is_prefix=True)
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


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_data(N_images_per_rank, ds):
    """
    Load intensity slices, reciprocal pixel position, index map
    Perform binning 
    """
    comm = contexts.comm
    rank = comm.rank
    size = comm.size

    # Load intensity slices, reciprocal pixel position, index map
    pixel_position_reciprocal = get_pixel_position_reciprocal(comm)
    pixel_index_map = get_pixel_index_map(comm)
    slices_ = get_slices(comm, N_images_per_rank, ds)
    N_images_local = slices_.shape[0]

    orientations_prior = get_orientations_prior(comm, N_images_per_rank)

    # Log mean image and saxs before binning
    mean_image = compute_mean_image(comm, slices_)
    if rank == (2 if settings.use_psana else 0):
        image.show_image(pixel_index_map, slices_[0], "image_0.png")
    if rank == 0:
        pixel_distance_reciprocal = prep.compute_pixel_distance(
            pixel_position_reciprocal)
        image.show_image(pixel_index_map, mean_image, "mean_image.png")
        #prep.export_saxs(pixel_distance_reciprocal, mean_image, "saxs.png")

    # Bin reciprocal position, reciprocal distance, index map, slices
    pixel_position_reciprocal = prep.binning_mean(pixel_position_reciprocal)
    pixel_index_map = prep.binning_index(pixel_index_map)
    pixel_distance_reciprocal = prep.compute_pixel_distance(
            pixel_position_reciprocal)
    slices_ = prep.binning_sum(slices_)

    # Log mean image and saxs after binning
    mean_image = compute_mean_image(comm, slices_)
    if rank == (2 if settings.use_psana else 0):
        image.show_image(pixel_index_map, slices_[0], "image_binned_0.png")
    if rank == 0:
        image.show_image(pixel_index_map, mean_image, "mean_image_binned.png")
        #prep.export_saxs(pixel_distance_reciprocal, mean_image,
        #                 "saxs_binned.png")

    return (pixel_position_reciprocal,
            pixel_distance_reciprocal,
            pixel_index_map,
            slices_,
            orientations_prior)
