from functools import wraps
import numpy  as np
import PyNVTX as nvtx

from pygion import task, Region, Partition, Tunable, R

from spinifel import SpinifelContexts


def gpu_task_wrapper(thunk):
    @wraps(thunk)
    def wrapper(*args, **kwargs):
        context = SpinifelContexts()
        if context.ctx is not None:
            context.ctx.push()
        try:
            return thunk(*args, **kwargs)
        finally:
            if context.ctx is not None:
                context.ctx.pop()
    return wrapper


@task(privileges=[R])
@nvtx.annotate("legion/utils.py", is_prefix=True)
def print_region(region):
    for field in region.keys():
        value = getattr(region, field).flatten()[0]
        print(f"{field}: {value}")



@nvtx.annotate("legion/utils.py", is_prefix=True)
def get_region_shape(region):
    return tuple(region.ispace.domain.extent)



@nvtx.annotate("legion/utils.py", is_prefix=True)
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
