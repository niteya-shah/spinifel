import h5py
import numpy as np
import os
import pygion
from pygion import task, Tunable, Partition, Region, WD

from spinifel import parms, prep


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


def get_data():
    pixel_position = get_pixel_position()
    pixel_index = get_pixel_index()
    slices, slices_p = get_slices()
    return (pixel_position, pixel_index, slices, slices_p)
