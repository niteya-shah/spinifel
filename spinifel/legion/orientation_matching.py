import numpy as np
import pygion
import socket
import PyNVTX as nvtx
from pygion import task, IndexLaunch, Partition, Region, RO, WD, Reduce, Tunable

from spinifel import parms
from spinifel.sequential.orientation_matching import slicing_and_match as sequential_match

from . import prep
from . import utils as lgutils


@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def setup_match():
    print("setup_match")
    N_orientations = parms.N_orientations
    N_images_per_rank = parms.N_images_per_rank
    N_batch_size = parms.N_batch_size
    fields_dict = {"quaternions": getattr(pygion, parms.data_type_str)}
    sec_shape = parms.quaternion_shape
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    print("N_procs =", N_procs)
    assert N_orientations % N_procs == 0, "N_orientations must be divisible by N_procs"
    N_orientations_per_rank = int(N_orientations / N_procs)
    print("N_orientations_per_rank =", N_orientations_per_rank)
    ref_orientations, ref_orientations_p = lgutils.create_distributed_region(
        N_orientations_per_rank, fields_dict, sec_shape)
    match_summary = Region((N_procs * N_orientations_per_rank, N_images_per_rank),
                {"euDist": pygion.float64})
    match_summary_p = Partition.equal(match_summary, (N_procs,))
    for i, ref_orientations_subr in enumerate(ref_orientations_p):
        prep.load_ref_orientations(ref_orientations_subr, i, N_orientations_per_rank, point=i)
    return ref_orientations, ref_orientations_p, match_summary, match_summary_p


@task(privileges=[RO, RO, WD, RO, WD, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_task(phased, slices, orientations, ref_orientations, match_summary, pixel_position, pixel_distance, rank):
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} starts Orientation Matching.", flush=True)
        print(f"{socket.gethostname()}:", end=" ", flush=False)
    orientations_matched_local, minDist_local = sequential_match(
        phased.ac, slices.data, pixel_position.reciprocal, pixel_distance.reciprocal, ref_orientations.quaternions)
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} finished Orientation Matching.", flush=True)
    match_summary.rank[0] = rank
    match_summary.orientations_matched[0] = orientations_matched_local
    match_summary.minDist[0] = minDist_local


@task(privileges=[RO, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def select_orientations(match_summary, orientations_selected):
    print('len(match_summary.rank) =', len(match_summary.rank))
    print('match_summary.rank =', match_summary.rank)
    print('match_summary.orientations_matched =', match_summary.orientations_matched)
    print('match_summary.minDist =', match_summary.minDist)
    index = np.argmin(match_summary.minDist)
    print('index =', index)
    orientations_matched = np.swapaxes(match_summary.orientations_matched, 0, 1)
    for i in range(len(index)):
        orientations_selected.quaternions[:] = match_summary.orientations_matched[i,index,:]
    return orientations_selected


@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match(phased, slices, slices_p, pixel_position, pixel_distance, orientations, orientations_p, ref_orientations, ref_orientations_p, match_summary, match_summary_p):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.

    N_ranks_per_node =  parms.N_ranks_per_node ##### TO-DO
    N_images_per_rank = parms.N_images_per_rank
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
        match_task(
            phased, slices_p[i], orientations_p[i/N_ranks_per_node], ref_orientations_p[i], match_summary_p[i],
            pixel_position, pixel_distance, i)

    for i in IndexLaunch([N_procs/N_ranks_per_node]):
        orientations_p[i] = select_orientations(match_summary, orientations_p[i])
    
    return orientations, orientations_p
