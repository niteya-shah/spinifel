def get_region_shape(region):
    return tuple(up-low+1 for low, up in zip(*region.ispace.bounds))
