import pygion
import numpy as np
import socket
import PyNVTX as nvtx
from pygion import task, RO, WD, IndexLaunch, Tunable, LayoutConstraint, SOA_C, SOA_F
from spinifel import settings, utils
from . import utils as lgutils
from . import prep as gprep

@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def create_orientations_rp(n_images_per_rank):
    # quaternions are always double precision
    orientations, orientations_p = lgutils.create_distributed_region(
        n_images_per_rank, {"quaternions": pygion.float64}, (4,)
    )
    return orientations, orientations_p

@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def create_min_dist_rp(n_images_per_rank, n_conf):
    if settings.use_single_prec:
        float_type = pygion.float32
    else:
        float_type = pygion.float64

    # number of partitions = N_procs * N_conformations
    n_parts = Tunable.select(Tunable.GLOBAL_PYS).get()*n_conf

    # min_dist
    min_dist, min_dist_p, min_dist_proc = lgutils.create_distributed_region_with_num_parts(
        n_images_per_rank,  n_parts, {"min_dist": float_type}, () )

    conf, conf_p = lgutils.create_distributed_region(
        n_images_per_rank, {"conf_id": pygion.int32}, ()
    )

    return min_dist, min_dist_p, min_dist_proc, conf, conf_p


# The reference orientations don't have to match exactly between ranks.
# Each rank aligns its own slices.
# We can call the sequential function on each rank, provided that the
# cost of generating the model_slices isn't prohibitive.
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match(
        phased, orientations_p, slices_p, n_images_per_rank,
        conf_id=None, ready_objs=None):

    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    multiple_conf = None
    if settings.N_conformations > 1:
        multiple_conf = conf_id
    for idx in range(N_procs):
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
        i = N_procs - idx - 1
        if ready_objs is not None:
            match_task(
                phased,
                orientations_p[i],
                slices_p[i],
                ready_objs[i],
                multiple_conf,
                point=i)
        else:
            match_task(
                phased,
                orientations_p[i],
                slices_p[i],
                None,
                multiple_conf,
                point=i)


@task(leaf=True, privileges=[RO("ac"), WD("quaternions"), RO("data")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_task(phased, orientations, slices, ready_obj, conf_idx):
    snm = None
    logger = None

    if ready_obj is not None:
        ready_obj = ready_obj.get()
    logger = gprep.all_objs["logger"]
    snm = gprep.all_objs["snm"]

    logger.log(f"{socket.gethostname()} starts Orientation Matching",level=1)
    orientations.quaternions[:] = snm.slicing_and_match(phased.ac)
    logger.log(f"{socket.gethostname()} finished Orientation Matching.",level=1)


@task(leaf=True, privileges=[RO("ac"), WD("quaternions"), RO("data"), WD("min_dist")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_task_conf(phased, orientations, slices, dist, conf_idx, ready_obj):
    snm = None
    logger = None
    if ready_obj is not None:
        ready_obj = ready_obj.get()
    logger = gprep.multiple_all_objs[conf_idx]["logger"]
    snm = gprep.multiple_all_objs[conf_idx]["snm"]
    logger.log(f"{socket.gethostname()} starts Orientation Matching",level=1)
    orientations.quaternions[:], dist.min_dist[:] = snm.slicing_and_match_with_min_dist(phased.ac)
    logger.log(f"{socket.gethostname()} finished Orientation Matching.",level=1)

# The reference orientations don't have to match exactly between ranks.
# Each rank aligns its own slices.
# We can call the sequential function on each rank, provided that the
# cost of generating the model_slices isn't prohibitive.
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_single_conf(
        phased, orientations_p, slices_p, dist_p, n_images_per_rank,
        conf_idx, num_conf, ready_objs=None):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    logger = gprep.multiple_all_objs[conf_idx]["logger"]
    for idx in range(N_procs):
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
        i = N_procs - idx - 1
        dist_idx = i*num_conf + conf_idx
        if ready_objs is not None:
            match_task_conf(
                phased,
                orientations_p[i],
                slices_p[i],
                dist_p[dist_idx],
                conf_idx,
                ready_objs[i],
                point=i)
        else:
            match_task_conf(
                phased,
                orientations_p[i],
                slices_p[i],
                dist_p[dist_idx],
                conf_idx,
                None,
                point=i)

# conf region contains the conformation id based on min distance
# for each diffraction image
@task(leaf=True, privileges=[RO("min_dist"), WD("conf_id")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def select_conf_task(dist_r, conf):
    a = settings.N_images_per_rank
    b = settings.N_conformations
    x = dist_r.min_dist.reshape(b,a)
    logger = gprep.multiple_all_objs[0]["logger"]
    logger.log(f"x = {x.shape}, {x.dtype}", level=2)
    logger.log(f"x = {x}", level=2)
    conf.conf_id[:] = np.argmin(x, axis=0)
    logger.log(f"conf_id = {conf.conf_id.shape}, {conf.conf_id.dtype}", level=2)
    logger.log(f"conf_ids = {conf.conf_id}", level=2)

#min_dist is a region -> N_conformations x N_images_per_rank x N_ranks
#conf_p is a region/partition -> for each N_images_per_rank return the index for the conformation it belongs to
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_conf(phased, orientations_p, slices_p, min_dist_p, min_dist_proc, conf_p, n_images_per_rank, ready_objs):
    for i in range(settings.N_conformations):
        match_single_conf(phased[i], orientations_p[i], slices_p, min_dist_p, n_images_per_rank, i, settings.N_conformations, ready_objs)

    # determine which conformation each diffraction pattern belongs to
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in range(N_procs):
        select_conf_task(min_dist_proc[i], conf_p[i], point=i)
