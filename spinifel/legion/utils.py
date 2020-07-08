import numpy as np
from pygion import Region, Partition, Tunable


def get_region_shape(region):
    return tuple(region.ispace.domain.extent)


def create_distributed_region(N_images_per_rank, fields_dict, sec_shape):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_images = N_procs * N_images_per_rank
    shape_total = (N_images,) + sec_shape
    shape_local = (N_images_per_rank,) + sec_shape
    region = Region(shape_total, fields_dict)
    region_p = Partition.restrict(
        region, [N_procs],
        N_images_per_rank * np.eye(len(shape_total), 1),
        shape_local)
    return region, region_p
