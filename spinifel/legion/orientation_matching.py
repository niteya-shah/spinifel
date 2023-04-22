import pygion

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
    # multiple conformations?
    mult_conf = False
    if conf_idx is not None:
        mult_conf = True
        logger = gprep.multiple_all_objs[conf_idx]["logger"]
        snm = gprep.multiple_all_objs[conf_idx]["snm"]
    else:
        logger = gprep.all_objs["logger"]
        snm = gprep.all_objs["snm"]

    logger.log(f"{socket.gethostname()} starts Orientation Matching",level=1)

    orientations.quaternions[:] = snm.slicing_and_match(phased.ac)
    logger.log(f"{socket.gethostname()} finished Orientation Matching.",level=1)


# The reference orientations don't have to match exactly
# between ranks.
# Each rank aligns its own slices.
# We can call the sequential function on each rank,
# provided that the
# cost of generating the model_slices isn't prohibitive.
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_conf(phased, orientations_p, slices_p, n_images_per_rank, ready_objs=None):
    is_ready = False
    if ready_objs != None:
        is_ready = True
    for i in range(settings.N_conformations):
        if is_ready:
            match(phased[i], orientations_p[i], slices_p, n_images_per_rank, i, None)
        else:
            match(phased[i], orientations_p[i], slices_p, n_images_per_rank, i, ready_objs[i])
