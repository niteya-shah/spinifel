import h5py
import numpy as np
import os
import pygion
from pygion import task, Tunable, Partition, Region, WD, RO, Reduce

from spinifel import parms, prep, image

from . import utils as lgutils


@task(privileges=[WD])
def load_pixel_position(pixel_position):
    prep.load_pixel_position_reciprocal(pixel_position.reciprocal)


def get_pixel_position():
    pixel_position_type = getattr(pygion, parms.pixel_position_type_str)
    pixel_position = Region(parms.pixel_position_shape,
                            {'reciprocal': pixel_position_type})
    load_pixel_position(pixel_position)
    return pixel_position


@task(privileges=[WD])
def load_pixel_index(pixel_index):
    prep.load_pixel_index_map(pixel_index.map)


def get_pixel_index():
    pixel_index_type = getattr(pygion, parms.pixel_index_type_str)
    pixel_index = Region(parms.pixel_index_shape,
                         {'map': pixel_index_type})
    load_pixel_index(pixel_index)
    return pixel_index


@task(privileges=[WD])
def load_slices(slices, rank, N_images_per_rank):
    i_start = rank * N_images_per_rank
    i_end = i_start + N_images_per_rank
    prep.load_slices(slices.data, i_start, i_end)


def get_slices():
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_images_per_rank = parms.N_images_per_rank
    N_images = N_procs * N_images_per_rank
    data_type = getattr(pygion, parms.data_type_str)
    data_shape_total = (N_images,) + parms.det_shape
    data_shape_local = (N_images_per_rank,) + parms.det_shape
    slices = Region(data_shape_total, {'data': data_type})
    slices_p = Partition.restrict(
        slices, [N_procs],
        N_images_per_rank * np.eye(len(data_shape_total), 1),
        data_shape_local)
    for i in range(N_procs):
        load_slices(slices_p[i], i, N_images_per_rank)
    return slices, slices_p


@task(privileges=[RO, Reduce('+')])
def reduce_mean_image(slices, mean_image):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    mean_image.data[:] += slices.data.mean(axis=0) / N_procs


def compute_mean_image(slices, slices_p):
    mean_image = Region(lgutils.get_region_shape(slices)[1:],
                        {'data': pygion.float32})
    pygion.fill(mean_image, 'data', 0.)
    for slices in slices_p:
        reduce_mean_image(slices, mean_image)
    return mean_image


@task(privileges=[RO, WD])
def calculate_pixel_distance(pixel_position, pixel_distance):
    pixel_distance.reciprocal[:] = prep.compute_pixel_distance(
            pixel_position.reciprocal)


def compute_pixel_distance(pixel_position):
    pixel_position_type = getattr(pygion, parms.pixel_position_type_str)
    pixel_distance = Region(lgutils.get_region_shape(pixel_position)[1:],
                            {'reciprocal': pixel_position_type})
    calculate_pixel_distance(pixel_position, pixel_distance)
    return pixel_distance


@task(privileges=[RO, WD])
def apply_pixel_position_binning(old_pixel_position, new_pixel_position):
    new_pixel_position.reciprocal[:] = prep.binning_mean(
        old_pixel_position.reciprocal)


def bin_pixel_position(old_pixel_position):
    pixel_position_type = getattr(pygion, parms.pixel_position_type_str)
    new_pixel_position = Region(parms.reduced_pixel_position_shape,
                                {'reciprocal': pixel_position_type})
    apply_pixel_position_binning(old_pixel_position, new_pixel_position)
    return new_pixel_position


@task(privileges=[RO, WD])
def apply_pixel_index_binning(old_pixel_index, new_pixel_index):
    new_pixel_index.map[:] = prep.binning_index(
        old_pixel_index.map)


def bin_pixel_index(old_pixel_index):
    pixel_index_type = getattr(pygion, parms.pixel_index_type_str)
    new_pixel_index = Region(parms.reduced_pixel_index_shape,
                             {'map': pixel_index_type})
    apply_pixel_index_binning(old_pixel_index, new_pixel_index)
    return new_pixel_index


@task(privileges=[RO, WD])
def apply_slices_binning(old_slices, new_slices):
    new_slices.data[:] = prep.binning_sum(old_slices.data)


def bin_slices(old_slices, old_slices_p):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_images_per_rank = parms.N_images_per_rank
    N_images = N_procs * N_images_per_rank
    data_type = getattr(pygion, parms.data_type_str)
    data_shape_total = (N_images,) + parms.reduced_det_shape
    data_shape_local = (N_images_per_rank,) + parms.reduced_det_shape
    new_slices = Region(data_shape_total, {'data': data_type})
    new_slices_p = Partition.restrict(
        new_slices, [N_procs],
        N_images_per_rank * np.eye(len(data_shape_total), 1),
        data_shape_local)
    for i in range(N_procs):
        apply_slices_binning(old_slices_p[i], new_slices_p[i])
    return new_slices, new_slices_p


@task(privileges=[RO, RO])
def show_image(pixel_index, images, image_index, name):
    image.show_image(pixel_index.map, images.data[image_index], name)


@task(privileges=[RO, RO])
def export_saxs(pixel_distance, mean_image, name):
    np.seterr(invalid='ignore')
    # Avoid warning in SAXS 0/0 division.
    # Legion seems to reset the global seterr from parms.py.
    prep.export_saxs(pixel_distance.reciprocal, mean_image.data, name)


def get_data():
    pixel_position = get_pixel_position()
    pixel_index = get_pixel_index()
    slices, slices_p = get_slices()
    mean_image = compute_mean_image(slices, slices_p)
    show_image(pixel_index, slices_p[0], 0, "image_0.png")
    show_image(pixel_index, mean_image, ..., "mean_image.png")
    pixel_distance = compute_pixel_distance(pixel_position)
    export_saxs(pixel_distance, mean_image, "saxs.png")
    pixel_position = bin_pixel_position(pixel_position)
    pixel_index = bin_pixel_index(pixel_index)
    slices, slices_p = bin_slices(slices, slices_p)
    mean_image = compute_mean_image(slices, slices_p)
    show_image(pixel_index, slices_p[0], 0, "image_binned_0.png")
    show_image(pixel_index, mean_image, ..., "mean_image_binned.png")
    pixel_distance = compute_pixel_distance(pixel_position)
    export_saxs(pixel_distance, mean_image, "saxs_binned.png")
    return (pixel_position, pixel_distance, pixel_index, slices, slices_p)
