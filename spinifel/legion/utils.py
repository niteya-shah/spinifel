import numpy  as np
import PyNVTX as nvtx

from pygion import task, Region, Partition, Tunable, R

from spinifel import SpinifelContexts



class GPUTaskWrapper(object):
    # Note: Can't use __slots__ for this class because __qualname__
    # conflicts with the class variable.
    def __init__(self, thunk):
        print(repr((thunk, thunk.__name__, thunk.__qualname__, thunk.__module__)))
        self.thunk = thunk
        self.__name__ = thunk.__name__
        self.__qualname__ = thunk.__qualname__
        self.__module__ = thunk.__module__
    def __call__(self, *args, **kwargs):
        context = SpinifelContexts()
        context.ctx.push()
        try:
            return self.thunk(*args, **kwargs)
        finally:
            context.ctx.pop()

def gpu_task_wrapper(thunk):
    return GPUTaskWrapper(thunk)



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

