import pygion
import numpy as np
import socket
import PyNVTX as nvtx
from pygion import task, RO, WD, IndexLaunch, Tunable, LayoutConstraint, SOA_C, SOA_F
from spinifel import settings, utils
from . import utils as lgutils
from . import prep as gprep
from .fsc import check_convergence_single_conf


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
        n_images_per_rank*n_conf, {"conf_id": pygion.float32}, ()
    )
    multiple_conf_regions = {}
    multiple_conf_regions["min_dist"] = min_dist
    multiple_conf_regions["min_dist_p"] = min_dist_p
    multiple_conf_regions["min_dist_proc"] = min_dist_proc
    multiple_conf_regions["conf"] = conf
    multiple_conf_regions["conf_p"] = conf_p
    return multiple_conf_regions

# initialize conf with random values
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
@lgutils.gpu_task_wrapper
@task(leaf=True, privileges=[WD])
def init_conf_task(conf, num_images, num_conf, mode):
    x = conf.conf_id.reshape(num_conf,num_images)
    # randomly select a particular conformation for each diffraction pattern
    # a = number of images
    arg_index = np.random.choice(num_conf,num_images)
    for i in range(num_conf):
        x[i] = np.where(arg_index==i, 1.0, 0.0)
    conf.conf_id[:] = x.reshape(num_conf*num_images)

# initialize conf with pre-determined values
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def init_conf_known(conf, num_images, num_conf, mode, n_procs):
    for i in range (n_procs):
        gprep.load_conformations_prior(conf[i], num_conf, i, settings.N_images_per_rank, point=i)

# initialize conf with random values
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def init_conf(conf_p, num_images):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    num_conf = settings.N_conformations
    mode = settings.conformation_mode
    # if single conformation - all images belong to it
    # if testing mode - all images belong to all conformations
    if num_conf == 1 or mode == "test_debug":
        for i in range (N_procs):
            pygion.fill(conf_p[i], "conf_id", 1.0)
            # support fsc_fraction_known_orientations = 1.0 only
            # only way to get deterministic results
            #    elif num_conf > 1 and settings.fsc_fraction_known_orientations == 1.0:
            #        init_conf_known(conf_p, num_images, num_conf, mode, N_procs)
            #    elif num_conf > 1:
            #        assert settings.fsc_fraction_known_orientations != 0.0:
    else:
        for i in range (N_procs):
            init_conf_task(conf_p[i], num_images, num_conf, mode, point=i)

# this needs to be updated with the right value
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def fill_min_dist(min_dist_p, conf_idx, num_conf):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in range (N_procs):
        dist_idx = i*num_conf + conf_idx
        pygion.fill(min_dist_p[dist_idx], "min_dist", 1000000000.0)

# The reference orientations don't have to match exactly between ranks.
# Each rank aligns its own slices.
# We can call the sequential function on each rank, provided that the
# cost of generating the model_slices isn't prohibitive.
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match(
        phased, orientations_p, slices_p, n_images_per_rank,ready_objs,stream=False):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for idx in range(N_procs):
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
        i = N_procs - idx - 1
        match_task(
            phased,
            orientations_p[i],
            slices_p[i],
            ready_objs[i],
            stream,
            point=i)

@task(leaf=True, privileges=[RO("ac"), WD("quaternions"), RO("data"), RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_task(phased, orientations, slices, ready_obj, stream=False):
    snm = None
    logger = None
    if stream:
        logger = gprep.multiple_all_objs[0]["logger"]
        snm = gprep.multiple_all_objs[0]["snm"]
    else:
        logger = gprep.all_objs["logger"]
        snm = gprep.all_objs["snm"]

    logger.log(f"{socket.gethostname()} starts Orientation Matching",level=1)
    orientations.quaternions[:] = snm.slicing_and_match(phased.ac)
    logger.log(f"{socket.gethostname()} finished Orientation Matching.",level=1)

@task(leaf=True, privileges=[RO("ac"), WD("quaternions"), RO("data"), WD("min_dist"), RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_task_conf(phased, orientations, slices, dist, ready_obj, conf_idx):
    snm = None
    logger = None
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
        conf_idx, num_conf, ready_objs):

    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    logger = gprep.multiple_all_objs[conf_idx]["logger"]
    for idx in range(N_procs):
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
        i = N_procs - idx - 1
        dist_idx = i*num_conf + conf_idx
        match_task_conf(
            phased,
            orientations_p[i],
            slices_p[i],
            dist_p[dist_idx],
            ready_objs[i],
            conf_idx,
            point=i)


@task(leaf=True, privileges=[RO("min_dist"), WD("conf_id")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def select_conf_task(dist_r, conf, mode):
    a = conf.ispace.domain.extent[0]//settings.N_conformations
    b = settings.N_conformations
    logger = gprep.multiple_all_objs[0]["logger"]
    snm = gprep.multiple_all_objs[0]["snm"]
    # shape of x -> [N_conformations,N_images_per_proc]
    x = dist_r.min_dist.reshape(b,a)
    logger.log(f"select_conf_task:x = {x.shape}, {x.dtype}", level=2)
    # shape of conf and dist_r -> [N_conformations*N_images_per_proc]
    logger.log(f"select_conf_task:dist_r = {dist_r.min_dist.shape}, {dist_r.min_dist.dtype}", level=2)
    logger.log(f"select_conf_task:conf_id = {conf.conf_id.shape}, {conf.conf_id.dtype}", level=2)
    # the higher the conf_id value, the less likely
    conf.conf_id[:] = snm.conformation_result(x,mode).reshape(a*b)

#min_dist is a region -> N_conformations x N_images_per_rank x N_ranks
#conf_p is a region/partition -> for each N_images_per_rank return the index for the conformation it belongs to
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_conf(phased, orientations_p, slices_p, min_dist_p, min_dist_proc, conf_p, n_images_per_rank, ready_objs, fsc):
    logger = gprep.multiple_all_objs[0]["logger"]
    # need to select_conf_task if N_conformations > 1
    if settings.N_conformations > 1:
        for i in range(settings.N_conformations):
            # check fsc future values -> if convergence has failed then continue with match
            if len(fsc) > 0 and check_convergence_single_conf(fsc[i]):
                logger.log(f"conformation {i} HAS converged in orientation_matching check")
                fill_min_dist(min_dist_p, i, settings.N_conformations)
            else:
                if len(fsc) > 0:
                    logger.log(f"conformation {i} has NOT converged in orientation_matching check")
                match_single_conf(phased[i], orientations_p[i], slices_p, min_dist_p, n_images_per_rank, i, settings.N_conformations, ready_objs)
        N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
        # TODO initialize min_dist_proc to zero for those conformations that are completed
        for i in range(N_procs):
            select_conf_task(min_dist_proc[i], conf_p[i], settings.conformation_mode, point=i)
    else:
        match(phased[0], orientations_p[0], slices_p, n_images_per_rank,ready_objs, True)

