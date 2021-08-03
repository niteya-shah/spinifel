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
    print(f"setup_match", flush=True)
    N_orientations = parms.N_orientations
    N_images_per_rank = parms.N_images_per_rank
    N_ranks_per_node =  parms.N_ranks_per_node
    fields_dict = {"quaternions": getattr(pygion, parms.data_type_str)}
    sec_shape = parms.quaternion_shape
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    print(f"N_procs = {N_procs}", flush=True)
    N_nodes = N_procs // N_ranks_per_node
    print(f"N_nodes = {N_nodes}", flush=True)
    assert N_orientations % N_ranks_per_node == 0, "N_orientations must be divisible by N_ranks_per_node"
    N_orientations_per_rank = int(N_orientations / N_ranks_per_node)
    print(f"N_orientations_per_rank = {N_orientations_per_rank}", flush=True)
    ref_orientations, ref_orientations_p = lgutils.create_distributed_region(
        N_orientations_per_rank, fields_dict, sec_shape)
    match_summary, match_summary_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    dist_summary = Region((N_procs * N_images_per_rank), {"minDist": pygion.float64})
    dist_summary_p = Partition.equal(dist_summary, (N_procs,))
    shape_total = (N_nodes * N_images_per_rank,) + sec_shape
    shape_local = (N_images_per_rank,) + sec_shape
    match_summary_p_nnodes = Partition.restrict(
        match_summary, [N_nodes],
        N_images_per_rank * np.eye(len(shape_total), 1),
        shape_local)
    dist_summary_p_nnodes = Partition.equal(dist_summary, (N_nodes,))
    for i, ref_orientations_subr in enumerate(ref_orientations_p):
        prep.load_ref_orientations(ref_orientations_subr, i, N_orientations_per_rank, point=i)
    print(f"passed setup_match", flush=True)
    return ref_orientations, ref_orientations_p, match_summary, match_summary_p, dist_summary, dist_summary_p


@task(privileges=[RO, RO, WD, RO, WD, WD, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_task(phased, slices, orientations, ref_orientations, match_summary, dist_summary, pixel_position, pixel_distance):
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} starts Orientation Matching.", flush=True)
        print(f"{socket.gethostname()}:", end=" ", flush=False)
    orientations_matched_local, minDist_local = sequential_match(
        phased.ac, slices.data, pixel_position.reciprocal, pixel_distance.reciprocal, ref_orientations.quaternions)
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} finished Orientation Matching.", flush=True)
    match_summary.quaternions[:] = orientations_matched_local
    dist_summary.minDist[:] = minDist_local
    print(f"passed match_task", flush=True)


@task(privileges=[RO, RO, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def select_orientations(match_summary, dist_summary, orientations_selected):
    print(f"match_summary.quaternions = {match_summary.quaternions}", flush=True)
    print(f"dist_summary.minDist = {dist_summary.minDist}", flush=True)
    index = np.argmin(dist_summary.minDist)
    print(f"index = {index}", flush=True)
    orientations_matched = np.swapaxes(match_summary.quaternions, 0, 1)
    for i in range(len(index)):
        orientations_selected.quaternions[:] = match_summary.quaternions[i,index,:]
    print(f"passed select_orientations", flush=True)
    return orientations_selected


@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match(phased, slices, slices_p, pixel_position, pixel_distance, orientations, orientations_p, ref_orientations, ref_orientations_p, match_summary, match_summary_p, dist_summary, dist_summary_p):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    N_ranks_per_node =  parms.N_ranks_per_node ##### TO-DO
    N_images_per_rank = parms.N_images_per_rank
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_nodes = N_procs // N_ranks_per_node
    curr_index = 0
    for i in IndexLaunch([N_procs]):
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
        j = curr_index // N_ranks_per_node
        print(f"i = {i}", flush=True)
        print(f"j = {j}", flush=True)
        match_task(
            phased, slices_p[i], orientations_p[j], ref_orientations_p[i], match_summary_p[i],
            pixel_position, pixel_distance, i)
        curr_index += 1
    for i in IndexLaunch([N_nodes]):
        orientations_p[i] = select_orientations(match_summary_p_nnodes[i], dist_summary_p_nnodes[i], orientations_p[i])
    print(f"passed match", flush=True)
    return orientations, orientations_p
