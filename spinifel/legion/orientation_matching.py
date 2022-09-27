import pygion

import socket
import PyNVTX as nvtx
from pygion import task, RO, WD, IndexLaunch, Tunable, LayoutConstraint, SOA_C, SOA_F
from spinifel import settings
from . import utils as lgutils
from . import prep as gprep

@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def create_orientations_rp(n_images_per_rank):
    if settings.use_single_prec:
        orientations, orientations_p = lgutils.create_distributed_region(
            n_images_per_rank, {"quaternions": pygion.float32}, (4,))
    else:
        orientations, orientations_p = lgutils.create_distributed_region(
            n_images_per_rank, {"quaternions": pygion.float64}, (4,))
    return orientations, orientations_p

@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match(phased, orientations_p, slices_p, n_images_per_rank):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for idx in range(N_procs):
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
        i=N_procs-idx-1
        match_task(
            phased, orientations_p[i], slices_p[i], point=i)

@task(leaf=True, privileges=[RO("ac"), WD("quaternions"), RO("data")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_task(phased, orientations, slices):
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} starts Orientation Matching.", flush=True)
    snm = gprep.all_objs['snm']
    orientations.quaternions[:] = snm.slicing_and_match(phased.ac)
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} finished Orientation Matching.", flush=True)
