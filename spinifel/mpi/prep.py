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
def get_slices(comm, N_images_per_rank):
    """Each rank loads intensity slices from input hdf5 file."""
    data_type = getattr(np, settings.data_type_str)
    slices_ = np.zeros((N_images_per_rank,) + settings.det_shape,
                       dtype=data_type)
    i_start = comm.rank * N_images_per_rank
    i_end = i_start + N_images_per_rank
    print(f"get_slices rank={comm.rank} st={i_start} en={i_end}",flush=True)
    prep.load_slices(slices_, i_start, i_end)
    return slices_


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def get_slices_and_pixel_info(N_images_per_rank, ds):
    """Each rank loads intensity slices and pixel parameters using psana2."""
    data_type = getattr(np, settings.data_type_str)
    slices_ = np.zeros((N_images_per_rank,) + settings.det_shape,
                       dtype=data_type)
    pixel_position_reciprocal = None
    pixel_index_map = None
    i = 0

    # TODO: Legion - we can make callback works for spinifel
    # ds.analyze(callback, N_images_per_rank)


    for run in ds.runs():
        # TODO: We will need all detnames below to be part of toml file.
        det = run.Detector("amopnccd")

        # Test data (amo06516 and xpptut15/3iyf) store run's related data in BeginRun.
        # In the real case, these data will come from calibration constant.
        # Note:
        # - For amo06516, pixel position reciprocal will be calculated (see
        #   below) using the photon energy value per event.
        # - The move axis is done so that the dimension xy (2d) or xyz (3d) 
        #   becomes the first axis.
        _pixel_index_map = run.beginruns[0].scan[0].raw.pixel_index_map
        pixel_index_map = np.moveaxis(_pixel_index_map[:], -1, 0)
        if run.expt == "xpptut15":
            _pixel_position_reciprocal = run.beginruns[0].scan[0].raw.pixel_position_reciprocal
            pixel_position_reciprocal = np.moveaxis(
                _pixel_position_reciprocal[:], -1, 0)
        elif run.expt == "amo06516":
            pixel_position = run.beginruns[0].scan[0].raw.pixel_position
        
        for evt in run.events():
            raw = det.raw.calib(evt)

            # Only need to do once for per-run variables
            if i == 0 and run.expt=="amo06516":
                photon_energy = det.raw.photon_energy(evt)
                
                # Calculate pixel position in reciprocal space
                from skopi.beam import convert
                from skopi.geometry import get_reciprocal_space_pixel_position
                wavelength = convert.photon_energy_to_wavelength(
                        photon_energy)
                wavevector = np.array([0, 0, 1.0 / wavelength]) # skopi convention
                _pixel_position_reciprocal = get_reciprocal_space_pixel_position(
                        pixel_position, wavevector)
                pixel_position_reciprocal = np.moveaxis(
                    _pixel_position_reciprocal[:], -1, 0)

            try:
                slices_[i] = raw
            except IndexError:
                # TODO: Find way to include rank no. in the message below
                raise RuntimeError(
                    f"received too many events.")
            i += 1
    
    pixel_info = {}
    pixel_info['pixel_position_reciprocal'] = pixel_position_reciprocal
    pixel_info['pixel_index_map'] = pixel_index_map
    return slices_[:i], pixel_info


@nvtx.annotate("mpi/prep.py", is_prefix=True)
def compute_mean_image(slices_):
    if not contexts.is_worker: return None
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


def show_image(image, ds, rank, slices_, pixel_index_map, 
        pixel_position_reciprocal, pixel_distance_reciprocal, mean_image,
        image_0_name, image_mean_name,
        image_sax_name):
    """Asks a unique worker rank to write sample images to disk."""
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
def get_data(N_images_per_rank, ds):
    """
    Load intensity slices, reciprocal pixel position, index map
    Perform binning 
    """
    comm = contexts.comm
    rank = comm.rank
    size = comm.size

    # Load intensity slices, reciprocal pixel position, index map
    if ds is None:
        pixel_position_reciprocal = get_pixel_position_reciprocal(comm)
        pixel_index_map = get_pixel_index_map(comm)
        slices_ = get_slices(comm, N_images_per_rank)
    else:
        slices_, pixel_info = get_slices_and_pixel_info(N_images_per_rank, ds)
        pixel_position_reciprocal = pixel_info['pixel_position_reciprocal']
        pixel_index_map = pixel_info['pixel_index_map']
    N_images_local = slices_.shape[0]

    return (pixel_position_reciprocal,
            pixel_index_map,
            slices_)


def bin_data(pixel_position_reciprocal, pixel_index_map, slices_):
    ## Bin reciprocal position, reciprocal distance, index map, slices
    pixel_position_reciprocal = prep.binning_mean(pixel_position_reciprocal)
    pixel_index_map = prep.binning_index(pixel_index_map)
    slices_ = prep.binning_sum(slices_)
    return (pixel_position_reciprocal,
            pixel_index_map,
            slices_)

