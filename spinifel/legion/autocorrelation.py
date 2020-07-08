import numpy as np
import pygion
from pygion import task, WD
from scipy.sparse.linalg import LinearOperator, cg

import pysingfel as ps

from spinifel import parms
from . import utils as lgutils


@task(privileges=[WD])
def gen_random_orientations(orientations, N_images_per_rank):
    orientations.quaternions[:] = ps.get_random_quat(N_images_per_rank)


def get_random_orientations():
    N_images_per_rank = parms.N_images_per_rank
    fields_dict = {"quaternions": pygion.float64}
    sec_shape = (4,)
    orientations, orientations_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    for orientations_subr in orientations_p:
        gen_random_orientations(orientations_subr, N_images_per_rank)
    return orientations, orientations_p


def solve_ac(generation,
             pixel_position,
             pixel_distance,
             slices,
             slices_p,
             orientations=None,
             orientations_p=None,
             ac_estimate=None):
    if orientations is None:
        orientations, orientations_p = get_random_orientations()
