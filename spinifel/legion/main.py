import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R, WD

from spinifel import parms

from .prep import get_data


@task(privileges=[R])
def print_region(region):
    for field in region.keys():
        value = getattr(region, field).flatten()[0]
        print(f"{field}: {value}")


@task(replicable=True)
def main():
    print("In Legion main", flush=True)

    (pixel_position,
     pixel_index) = get_data()

    print_region(pixel_position)
    print_region(pixel_index)
