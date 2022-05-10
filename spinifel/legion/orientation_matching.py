import pygion
import socket
import PyNVTX as nvtx
from pygion import task, RO, WD, IndexLaunch, Tunable

from spinifel import settings
from spinifel.sequential.orientation_matching import slicing_and_match as sequential_match

from . import utils as lgutils



@task(privileges=[
    RO("ac"), RO("data"), WD("quaternions"), RO("reciprocal"), RO("reciprocal")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_task(phased, slices, orientations, pixel_position, pixel_distance, order):
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} starts Orientation Matching.", flush=True)
        print(f"{socket.gethostname()}:", end=" ", flush=False)
    orientations.quaternions[:] = sequential_match(
        phased.ac, slices.data, pixel_position.reciprocal, pixel_distance.reciprocal, order)
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} finished Orientation Matching.", flush=True)



@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match(phased, slices, slices_p, pixel_position, pixel_distance, order):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    orientations, orientations_p = lgutils.create_distributed_region(
        settings.N_images_per_rank, {"quaternions": pygion.float32}, (4,))

    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
        match_task(
            phased, slices_p[i], orientations_p[i],
            pixel_position, pixel_distance, order)

    return orientations, orientations_p
