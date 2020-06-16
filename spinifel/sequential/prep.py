import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from spinifel import parms, prep, image


def get_pixel_position_reciprocal():
    pixel_position_type = getattr(np, parms.pixel_position_type_str)
    pixel_position_reciprocal = np.zeros(parms.pixel_position_shape,
                                         dtype=pixel_position_type)
    prep.load_pixel_position_reciprocal(pixel_position_reciprocal)
    return pixel_position_reciprocal


def get_pixel_index_map():
    pixel_index_type = getattr(np, parms.pixel_index_type_str)
    pixel_index_map = np.zeros(parms.pixel_index_shape,
                               dtype=pixel_index_type)
    prep.load_pixel_index_map(pixel_index_map)
    return pixel_index_map


def get_slices(N_images):
    data_type = getattr(np, parms.data_type_str)
    slices_ = np.zeros((N_images,) + parms.det_shape,
                       dtype=data_type)
    prep.load_slices(slices_, 0, N_images)
    return slices_


def get_data(N_images):
    pixel_position_reciprocal = get_pixel_position_reciprocal()
    pixel_index_map = get_pixel_index_map()

    slices_ = get_slices(N_images)
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
