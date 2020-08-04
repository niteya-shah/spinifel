import pygion
from pygion import task, RO, WD

from spinifel import parms
from spinifel.sequential.orientation_matching import match as sequential_match

from . import utils as lgutils


@task(privileges=[
    RO("ac"), RO("data"), WD("quaternions"), RO("reciprocal"), RO("reciprocal")])
def match_task(phased, slices, orientations, pixel_position, pixel_distance):
    orientations.quaternions[:] = sequential_match(
        phased.ac, slices.data, pixel_position.reciprocal, pixel_distance.reciprocal)


def match(phased, slices, slices_p, pixel_position, pixel_distance):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    orientations, orientations_p = lgutils.create_distributed_region(
        parms.N_images_per_rank, {"quaternions": pygion.float32}, (4,))

    for orientations_subr, slices_subr in zip(orientations_p, slices_p):
        match_task(
            phased, slices_subr, orientations_subr, pixel_position, pixel_distance)

    return orientations, orientations_p
