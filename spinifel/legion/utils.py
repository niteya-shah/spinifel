from functools import wraps
import numpy as np
import PyNVTX as nvtx
from pygion import task, Region, Partition, Tunable, R, Ipartition, WD, Ispace, fill, execution_fence

from spinifel import SpinifelContexts, settings
from . import utils as lgutils


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
@lgutils.gpu_task_wrapper
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
        region, [N_procs], N_images_per_rank * np.eye(len(shape_total), 1), shape_local
    )
    return region, region_p

# returns 2 partitions -> one based on N_parts and another based on N_procs
@nvtx.annotate("legion/utils.py", is_prefix=True)
def create_distributed_region_with_num_parts(N_images_per_rank, N_parts, fields_dict, sec_shape):
    N_images = N_parts * N_images_per_rank
    shape_total = (N_images,) + sec_shape
    shape_local = (N_images_per_rank,) + sec_shape
    region = Region(shape_total, fields_dict)
    region_p = Partition.restrict(
        region, [N_parts], N_images_per_rank * np.eye(len(shape_total), 1),
        shape_local
    )
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    region_p2 = Partition.equal(region, N_procs)
    return region, region_p, region_p2


@nvtx.annotate("legion/utils.py", is_prefix=True)
def create_partition_with_offset(
    region, secShape, nImagesPerRank, maxImagesPerRank, stride, nPoints, offset
):
    nImages = nPoints * maxImagesPerRank
    shape_total = (nImages,) + secShape
    shape_local = (nImagesPerRank,) + secShape
    if offset == 0:
        region_p = Partition.restrict(
            region, [nPoints], stride * np.eye(len(shape_total), 1), shape_local
        )
        return region_p
    pending_p = Ipartition.pending(region.ispace, nPoints)
    for i in range(nPoints):
        ispace = []
        startpoint = (maxImagesPerRank * i + offset,) + (0, 0, 0)
        ispace.append(Ispace(shape_local, startpoint))
        pending_p.union([i], ispace)
        region_p = Partition(region, pending_p)
    return region_p


@nvtx.annotate("legion/utils.py", is_prefix=True)
def dump_single_partition(p):
    for j, p2 in enumerate(p):
        print(f" init_parts p2[{j}] = {p2.ispace.bounds}", flush=True)


@nvtx.annotate("legion/utils.py", is_prefix=True)
def dump_partitions(p):
    for i, p1 in enumerate(p):
        p1 = p[i]
        print(f"--------partition {i}-------", flush=True)
        dump_single_partition(p1)


# create a set of partitions to be used for filling in subregions
@nvtx.annotate("legion/utils.py", is_prefix=True)
def init_partitions(slices, nPoints, batchSize, maxBatchSize, curBatchSize, secShape):
    p = []
    # assume divisible by batchSize
    assert maxBatchSize % batchSize == 0
    num_parts = maxBatchSize // batchSize
    for i in range(num_parts):
        offset = curBatchSize
        p1 = create_partition_with_offset(
            slices, secShape, batchSize, maxBatchSize, maxBatchSize, nPoints, offset
        )
        curBatchSize = curBatchSize + batchSize
        p.append(p1)
    return p


def create_region(shape, fieldsDict):
    r = Region(shape, fieldsDict)
    return r


def create_fill_region(shape, fieldsDict, val):
    r = create_region(shape, fieldsDict)
    for field_name in fieldsDict.keys():
        fill(r, field_name, val)
    return r


@task(inner=True, privileges=[WD])
@lgutils.gpu_task_wrapper
def fill_region_task(merged, val):
    for field_name in merged.keys():
        fill(merged, field_name, val)


def fill_region(r, val):
    fill_region_task(r, val)


# create a region containing maxImagesPerRank*nPoints*secShape
def create_max_region(maxImagesPerRank, fieldsDict, secShape, nPoints):
    N_images = nPoints * maxImagesPerRank
    shape_total = (N_images,) + secShape
    region = Region(shape_total, fieldsDict)

    if settings.verbosity > 2:
        print(
            f" region = {region.ispace.bounds}, max_images_per_rank = {maxImagesPerRank}, n_points={nPoints}",
            flush=True,
        )
    return region


# union partitions with stride
def union_partitions_with_stride(
    region, secShape, nImagesPerRank, maxImagesPerRank, stride, nPoints
):
    nImages = nPoints * maxImagesPerRank
    shape_total = (nImages,) + secShape
    shape_local = (nImagesPerRank,) + secShape
    region_p = Partition.restrict(
        region, [nPoints], stride * np.eye(len(shape_total), 1), shape_local
    )

    if settings.verbosity > 3:
        for i in range(nPoints):
            print(
                f"Union: region_p[{i}] = {region_p[i].ispace.domain.extent}, {region_p[i].ispace.bounds}",
                flush=True,
            )

    return region_p


@nvtx.annotate("legion/utils.py", is_prefix=True)
def create_distributed_region_procs(fields_dict, sec_shape):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    #shape_total = (N_procs,) + sec_shape # 2,...
    shape_total = (N_procs*sec_shape[0],) + sec_shape[1:] # 2,...
    region = Region(shape_total, fields_dict)
    region_p = Partition.restrict(
        region, [N_procs], sec_shape[0]*np.eye(len(shape_total), 1), sec_shape
    )
    if settings.verbosity > 3:
        for i in range(N_procs):
            print(
                f"distributed_region_procs: region_p[{i}] = {region_p[i].ispace.domain.extent}, {region_p[i].ispace.bounds}",
                flush=True,
            )
    return region, region_p

