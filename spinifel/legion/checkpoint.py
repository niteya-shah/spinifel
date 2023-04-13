import numpy as np
import PyNVTX as nvtx
import os
import pygion
from pygion import task, Partition, Region, Tunable, WD, RO
from spinifel import settings, utils, contexts, checkpoint
from . import utils as lgutils


@task(privileges=[WD("ac", "support_", "rho_"), WD("quaternions")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/checkpoint.py", is_prefix=True)
def checkpoint_load_task(phased, orientations, out_dir, load_gen, tag_gen):
    logger = utils.Logger(True, settings)
    logger.log(
        f"Loading checkpoint: {checkpoint.generate_checkpoint_name(out_dir, load_gen, tag_gen)}",
        level=1,
    )
    myRes = checkpoint.load_checkpoint(out_dir, load_gen, tag_gen)
    # Unpack dictionary
    phased.ac[:] = myRes["ac_phased"]
    phased.support_[:] = myRes["support_"]
    phased.rho_[:] = myRes["rho_"]
    orientations.quaternions[:] = myRes["orientations"]


""" Create and Fill Regions [phased:{ac,support_,rho}],
                            [orientations:{quaternions}]
"""


def load_checkpoint(outdir: str, gen_num: int, tag=""):
    phased = Region(
        (settings.M,) * 3,
        {"ac": pygion.float32, "support_": pygion.float32, "rho_": pygion.float32},
    )

    # setup the orientation region
    N_images_per_rank = settings.N_images_per_rank
    fields_dict = {"quaternions": pygion.float32}
    sec_shape = (4,)
    orientations, orientations_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape
    )

    checkpoint_load_task(phased, orientations, outdir, gen_num, tag)
    return phased, orientations, orientations_p


""" 
Save pixel_position_reciprocal, pixel_distance_reciprocal
Save Regions [slices:{data}],
             [solved:{ac}]
"""


@task(privileges=[RO("data"), RO("ac"), RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/checkpoint.py", is_prefix=True)
def save_checkpoint_solve_ac(
    slices, solved, pixel_position, pixel_distance, out_dir: str, gen_num: int, tag=""
):
    # Pack dictionary
    myRes = {
        "pixel_position_reciprocal": pixel_position.reciprocal,
        "pixel_distance_reciprocal": pixel_distance.reciprocal,
        "slices_": slices.data,
        "ac": solved.ac,
    }
    checkpoint.save_checkpoint(myRes, out_dir, gen_num, tag)


""" Save Regions [solved:{ac}:float32],
                 [phased:{ac,support_,rho_}]
"""


@task(privileges=[RO("ac"), RO("ac"), RO("support_"), RO("rho_")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/checkpoint.py", is_prefix=True)
def save_checkpoint_phase(solved, phased, out_dir: str, gen_num: int, tag=""):
    # Pack dictionary
    myRes = {
        "ac": solved.ac,
        "ac_phased": phased.ac,
        "support_": phased.support_,
        "rho_": phased.rho_,
    }
    checkpoint.save_checkpoint(myRes, out_dir, gen_num, tag)


""" Save Regions [slices:{data}:float32],
                 [phased:{ac,support_,rho_}]: what about support/rho?
                 [orientations:{quaternions}]
                 [pixel_position:{reciprocal}]
                 [pixel_distance:{reciprocal}]
"""


@task(privileges=[RO("data"), RO("ac"), RO("quaternions"), RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/checkpoint.py", is_prefix=True)
def save_checkpoint_match(
    slices,
    phased,
    orientations,
    pixel_position,
    pixel_distance,
    out_dir: str,
    gen_num: int,
    tag="",
):
    # Pack dictionary
    myRes = {
        "ac_phased": phased.ac,
        "slices_": slices.data,
        "pixel_position_reciprocal": pixel_position.reciprocal,
        "pixel_distance_reciprocal": pixel_distance.reciprocal,
        "orientations": orientations.quaternions,
    }
    checkpoint.save_checkpoint(myRes, out_dir, gen_num, tag)


""" Save Regions [solved:{ac}:float32],
                 [prev_phased:{prev_rho_}]:
                 [phased:{ac,support_,rho_}]:

"""


@task(privileges=[RO("ac"), RO("prev_rho_"), RO("ac", "support_", "rho_")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/checkpoint.py", is_prefix=True)
def save_checkpoint_phase_prev(
    solved, prev_phased, phased, prev_support, out_dir: str, gen_num: int, tag=""
):
    myRes = {
        "ac": solved.ac,
        "prev_support_": prev_support,
        "prev_rho_": prev_phased.prev_rho_,
        "ac_phased": phased.ac,
        "support_": phased.support_,
        "rho_": phased.rho_,
    }

    checkpoint.save_checkpoint(myRes, out_dir, gen_num, tag)


""" Save Regions [phased:{ac,support_,rho_}]
                 [orientations:{quaternions}]
"""


@task(privileges=[RO("ac", "support_", "rho_"), RO("quaternions")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/checkpoint.py", is_prefix=True)
def save_checkpoint(phased, orientations, out_dir: str, gen_num: int, tag=""):
    # Pack dictionary
    myRes = {
        "ac_phased": phased.ac,
        "support_": phased.support_,
        "rho_": phased.rho_,
        "orientations": orientations.quaternions,
    }
    checkpoint.save_checkpoint(myRes, out_dir, gen_num, tag)
