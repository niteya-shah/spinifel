import h5py
import matplotlib.pyplot as plt
import numpy             as np
import PyNVTX            as nvtx

from matplotlib.colors import LogNorm

from spinifel import parms, prep, image



@nvtx.annotate("sequential/prep.py", is_prefix=True)
def get_pixel_position_reciprocal():
    """
    Return pixel positions in reciprocal space.
    """
    pixel_position_type = getattr(np, parms.pixel_position_type_str)
    pixel_position_reciprocal = np.zeros(parms.pixel_position_shape,
                                         dtype=pixel_position_type)
    prep.load_pixel_position_reciprocal(pixel_position_reciprocal)
    return pixel_position_reciprocal



@nvtx.annotate("sequential/prep.py", is_prefix=True)
def get_pixel_index_map():
    """
    Return pixel coordinates indexes from psana geometry.
    """
    pixel_index_type = getattr(np, parms.pixel_index_type_str)
    pixel_index_map = np.zeros(parms.pixel_index_shape,
                               dtype=pixel_index_type)
    prep.load_pixel_index_map(pixel_index_map)
    return pixel_index_map



@nvtx.annotate("sequential/prep.py", is_prefix=True)
def get_slices(N_images, ds):
    """
    Return data images.

    :param N_images: number of data images
    :param ds: data source
    :return slices_: data images
    """
    data_type = getattr(np, parms.data_type_str)
    slices_ = np.zeros((N_images,) + parms.det_shape,
                       dtype=data_type)
    if ds is None:
        prep.load_slices(slices_, 0, N_images)
    else:
        i = 0
        for run in ds.runs():
            for nevt, evt in enumerate(run.events()):
                raw = evt._dgrams[0].pnccdBack[0].raw
                slices_[i] = raw.image
                i += 1
                if i >= N_images:
                    return slices_
    return slices_



@nvtx.annotate("sequential/prep.py", is_prefix=True)
def get_data(N_images, ds):
    """
    Return pre-processed data for running M-TIP.    

    :param N_images: number of data images
    :param ds: data source
    :return pixel_position_reciprocal: pixel positions in reciprocal space
    :return pixel_distance_reciprocal: pixel distance in reciprocal space
    :return pixel_index_map: pixel coordinates indexes from psana geometry
    :return slices_: data images
    """
    pixel_position_reciprocal = get_pixel_position_reciprocal()
    pixel_index_map = get_pixel_index_map()

    slices_ = get_slices(N_images, ds)
    mean_image = slices_.mean(axis=0)

    image.show_image(pixel_index_map, slices_[0], "image_0.png")
    image.show_image(pixel_index_map, mean_image, "mean_image.png")

    pixel_distance_reciprocal = prep.compute_pixel_distance(
        pixel_position_reciprocal)
    prep.export_saxs(pixel_distance_reciprocal, mean_image, "saxs.png")

    pixel_position_reciprocal = prep.binning_mean(pixel_position_reciprocal)
    pixel_index_map = prep.binning_index(pixel_index_map)
    slices_ = prep.binning_sum(slices_)
    mean_image = slices_.mean(axis=0)

    image.show_image(pixel_index_map, slices_[0], "image_binned_0.png")
    image.show_image(pixel_index_map, mean_image, "mean_image_binned.png")

    pixel_distance_reciprocal = prep.compute_pixel_distance(
        pixel_position_reciprocal)
    prep.export_saxs(pixel_distance_reciprocal, mean_image, "saxs_binned.png")

    return (pixel_position_reciprocal,
            pixel_distance_reciprocal,
            pixel_index_map,
            slices_)
