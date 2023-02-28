import h5py
import matplotlib.pyplot as plt
import numpy as np
import PyNVTX as nvtx
from matplotlib.colors import LogNorm
from mpi4py import MPI

from spinifel import settings, prep, image, contexts
from spinifel.prep import load_pixel_position_reciprocal_psana


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_pixel_position_reciprocal():
    """Rank0 broadcast pixel reciprocal positions from input file."""
    comm = contexts.comm
    pixel_position_type = getattr(np, settings.pixel_position_type_str)
    pixel_position_reciprocal = np.zeros(
        settings.pixel_position_shape, dtype=pixel_position_type
    )
    if comm.rank == 0:
        prep.load_pixel_position_reciprocal(pixel_position_reciprocal)
    comm.Bcast(pixel_position_reciprocal, root=0)
    return pixel_position_reciprocal


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_pixel_index_map():
    """Rank0 broadcast pixel index map from input file."""
    comm = contexts.comm
    pixel_index_type = getattr(np, settings.pixel_index_type_str)
    pixel_index_map = np.zeros(settings.pixel_index_shape, dtype=pixel_index_type)
    if comm.rank == 0:
        prep.load_pixel_index_map(pixel_index_map)
    comm.Bcast(pixel_index_map, root=0)
    return pixel_index_map


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_orientations_prior(comm, N_images_per_rank):
    """Each rank loads correct orientations for data images from input file."""
    data_type = getattr(np, settings.data_type_str)
    orientations_prior = np.zeros((N_images_per_rank, 4), dtype=data_type)
    i_start = comm.rank * N_images_per_rank
    i_end = i_start + N_images_per_rank
    print(f"get_orientations_prior rank={comm.rank} st={i_start} en={i_end}")
    prep.load_orientations_prior(orientations_prior, i_start, i_end)
    return orientations_prior


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_slices(comm, N_images_per_rank):
    """Each rank loads intensity slices from input hdf5 file."""
    data_type = getattr(np, settings.data_type_str)
    slices_ = np.zeros((N_images_per_rank,) + settings.det_shape, dtype=data_type)
    i_start = comm.rank * N_images_per_rank
    i_end = i_start + N_images_per_rank
    print(f"get_slices rank={comm.rank} st={i_start} en={i_end}", flush=True)
    prep.load_slices(slices_, i_start, i_end)
    return slices_


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def compute_mean_image(slices_):
    if not contexts.is_worker:
        return None
    comm = contexts.comm_compute

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


def show_image(
    image,
    ds,
    rank,
    slices_,
    pixel_index_map,
    pixel_position_reciprocal,
    pixel_distance_reciprocal,
    mean_image,
    image_0_name,
    image_mean_name,
    image_sax_name,
):
    """Asks a unique worker rank to write sample images to disk."""
    if settings.show_image:
        show = False
        if ds is None:
            if rank == 0:
                show = True
        else:
            if ds.unique_user_rank():
                show = True

        if show:
            image.show_image(pixel_index_map, slices_[0], image_0_name)
            image.show_image(pixel_index_map, mean_image, image_mean_name)
            prep.export_saxs(pixel_distance_reciprocal, mean_image, image_sax_name)


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_data(N_images_per_rank):
    """
    Load intensity slices
    """
    slices_ = get_slices(contexts.comm, N_images_per_rank)
    return slices_


def get_pixel_info(run=None):
    """
    Load reciprocal pixel position, index map from hdf5 or gets it from
    BeginRun transition from xtc2.

    Note that pixel_position_reciprocal is converted to the expected shape
    in get_pixel_position_reciprocal 
    """
    # Pixel position in xtc2 file is used to calculate the pixel position
    # reciprocal with the given per-shot wavelength.
    pixel_position = None
    if run is None:
        pixel_position_reciprocal = get_pixel_position_reciprocal()
        pixel_index_map = get_pixel_index_map()
    else:
        det = run.Detector("amopnccd")
        
        pixel_position_reciprocal = np.zeros((3,) + settings.reduced_det_shape)
        if hasattr(run.beginruns[0].scan[0].raw, "pixel_position_reciprocal"):
            load_pixel_position_reciprocal_psana(run, pixel_position_reciprocal)
        
        _pixel_index_map = run.beginruns[0].scan[0].raw.pixel_index_map
        pixel_index_map = np.moveaxis(_pixel_index_map[:], -1, 0)

        if hasattr(run.beginruns[0].scan[0].raw, "pixel_position"):
            pixel_position = run.beginruns[0].scan[0].raw.pixel_position


    if settings.use_single_prec:
        pixel_position_reciprocal = pixel_position_reciprocal.astype(np.float32)

    return (pixel_position_reciprocal, pixel_index_map, pixel_position)


def bin_data(pixel_position_reciprocal=None, pixel_index_map=None, slices_=None):
    ## Bin reciprocal position, reciprocal distance, index map, slices
    if pixel_position_reciprocal is not None:
        pixel_position_reciprocal = prep.binning_mean(pixel_position_reciprocal)
    if pixel_index_map is not None:
        pixel_index_map = prep.binning_index(pixel_index_map)
    if slices_ is not None:
        slices_ = prep.binning_sum(slices_)
    return (pixel_position_reciprocal, pixel_index_map, slices_)
