import h5py
import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R, WD

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


def get_data():
    pixel_position = get_pixel_position()
    pixel_index = get_pixel_index()
    return (pixel_position, pixel_index)
