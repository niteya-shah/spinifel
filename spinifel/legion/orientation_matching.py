import numpy as np
import skopi as skp
import pygion
import socket
import PyNVTX as nvtx
from pygion import task, IndexLaunch, Partition, Region, RO, WD, Reduce, Tunable
from spinifel import parms, utils
from spinifel.sequential.orientation_matching import slicing_and_match as sequential_match
from . import prep
from . import utils as lgutils


@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def gen_nonuniform_batch(quaternions, H_, K_, L_, pixel_position, reciprocal_extent):
    N_orientations = parms.N_orientations
    N_ranks_per_node = parms.N_ranks_per_node
    assert N_orientations % N_ranks_per_node == 0, "N_orientations must be divisible by N_ranks_per_node"
    N_orientations_per_rank = int(N_orientations / N_ranks_per_node)
    print(f"N_orientations_per_rank = {N_orientations_per_rank}", flush=True)
    N_batch_size = parms.N_batch_size
    pixel_position_rp_c = np.array(pixel_position.reciprocal, copy=False, order='C')
    ref_rotmat = np.array([np.linalg.inv(skp.quaternion2rot3d(quat)) for quat in quaternions])
    for i in range(N_orientations_per_rank // N_batch_size):
        st = i * N_batch_size
        en = st + N_batch_size
        H, K, L = np.einsum("ijk,klmn->jilmn", ref_rotmat[st:en], pixel_position_rp_c)
        H_[i,:] = H.flatten() / reciprocal_extent * np.pi / parms.oversampling
        K_[i,:] = K.flatten() / reciprocal_extent * np.pi / parms.oversampling
        L_[i,:] = L.flatten() / reciprocal_extent * np.pi / parms.oversampling


@task(privileges=[RO, WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def fill_nonuniform_batch_task(ref_orientations, nonuniform_batch, pixel_position, reciprocal_extent):
    gen_nonuniform_batch(ref_orientations.quaternions, nonuniform_batch.H_, nonuniform_batch.K_, nonuniform_batch.L_, pixel_position, reciprocal_extent)


@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def setup_match(pixel_position, pixel_distance):
    N_orientations = parms.N_orientations
    N_images_per_rank = parms.N_images_per_rank
    N_ranks_per_node = parms.N_ranks_per_node
    N_pixels = utils.prod(parms.reduced_det_shape)
    N_batch_size = parms.N_batch_size
    reciprocal_extent = pixel_distance.reciprocal.max()
    fields_dict = {"quaternions": getattr(pygion, parms.data_type_str)}
    sec_shape = parms.quaternion_shape
    fields_dict_prime = {"H_": pygion.float64, "K_": pygion.float64,
                   "L_": pygion.float64}
    N_batch = N_pixels * N_batch_size
    sec_prime_shape = (N_batch,)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    print(f"N_procs = {N_procs}", flush=True)
    N_nodes = N_procs // N_ranks_per_node
    print(f"N_nodes = {N_nodes}", flush=True)
    assert N_orientations % N_ranks_per_node == 0, "N_orientations must be divisible by N_ranks_per_node"
    N_orientations_per_rank = int(N_orientations / N_ranks_per_node)
    print(f"N_orientations_per_rank = {N_orientations_per_rank}", flush=True)
    ref_orientations, ref_orientations_p = lgutils.create_distributed_region(
        N_orientations_per_rank, fields_dict, sec_shape)
    nonuniform_batch, nonuniform_batch_p = lgutils.create_distributed_region(
        N_orientations_per_rank // N_batch_size, fields_dict_prime, sec_prime_shape)
    match_summary, match_summary_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    dist_summary = Region((N_procs * N_images_per_rank), {"minDist": pygion.float64})
    dist_summary_p = Partition.equal(dist_summary, (N_procs,))
    shape_total = (N_procs * N_images_per_rank,) + sec_shape
    local_procs = N_procs // N_nodes
    shape_local = (local_procs * N_images_per_rank,) + sec_shape
    match_summary_p_nnodes = Partition.restrict(
        match_summary, [N_nodes],
        local_procs*N_images_per_rank * np.eye(len(shape_total), 1),
        shape_local)
    dist_summary_p_nnodes = Partition.equal(dist_summary, (N_nodes,))
    for i, ref_orientations_subr in enumerate(ref_orientations_p):
        prep.load_ref_orientations(ref_orientations_subr, i, N_orientations_per_rank, point=i)
    for i in IndexLaunch([N_procs]):
        fill_nonuniform_batch_task(ref_orientations_p[i], nonuniform_batch_p[i], pixel_position, reciprocal_extent)
    return ref_orientations, ref_orientations_p, nonuniform_batch, nonuniform_batch_p, match_summary, match_summary_p, dist_summary, dist_summary_p, match_summary_p_nnodes, dist_summary_p_nnodes


@task(privileges=[RO, RO, RO, WD, WD, RO, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match_task(phased, slices, ref_orientations, match_summary, dist_summary, nonuniform_batch, pixel_position, pixel_distance):
    print(f"in match_task, match_summary.quaternions.shape = {match_summary.quaternions.shape}", flush=True)
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} starts Orientation Matching.", flush=True)
        print(f"{socket.gethostname()}:", end=" ", flush=False)
    orientations_matched_local, minDist_local, H_, K_, L_ = sequential_match(
        phased.ac, slices.data, pixel_position.reciprocal, pixel_distance.reciprocal, ref_orientations.quaternions, nonuniform_batch.H_, nonuniform_batch.K_, nonuniform_batch.L_)
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} finished Orientation Matching.", flush=True)
    print(f"orientations_matched_local.shape = {orientations_matched_local.shape}", flush=True)
    print(f"minDist_local.shape = {minDist_local.shape}", flush=True)
    match_summary.quaternions[:] = orientations_matched_local
    dist_summary.minDist[:] = minDist_local
    print(f"match_summary.quaternions[:].shape = {match_summary.quaternions[:].shape}", flush=True)
    print(f"dist_summary.minDist[:].shape = {dist_summary.minDist[:].shape}", flush=True)


@task(privileges=[RO, RO, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def select_orientations(match_summary, dist_summary, orientations_selected):
    N_images_per_rank = parms.N_images_per_rank
    N_ranks_per_node = parms.N_ranks_per_node
    print(f"match_summary.quaternions.shape = {match_summary.quaternions.shape}", flush=True)
    print(f"dist_summary.minDist.shape = {dist_summary.minDist.shape}", flush=True)
    minDists = np.reshape(dist_summary.minDist, (N_ranks_per_node, N_images_per_rank), order='C')
    print(f"minDists shape = {minDists.shape}", flush=True)
    index = np.argmin(minDists, axis=0) 
    print(f"index.shape = {index.shape}", flush=True)
    orientations_matched = np.reshape(match_summary.quaternions, (N_ranks_per_node, N_images_per_rank, 4), order='C')
    print(f"orientations_matched = {orientations_matched.shape}")
    for i in range(len(index)):
        val = index[i]
        orientations_selected.quaternions[i,:] = orientations_matched[val,i,:]


@nvtx.annotate("legion/orientation_matching.py", is_prefix=True)
def match(phased, slices, slices_p, pixel_position, pixel_distance, orientations, orientations_p, ref_orientations, ref_orientations_p, nonuniform_batch, nonuniform_batch_p, match_summary, match_summary_p, dist_summary, dist_summary_p, match_summary_p_nnodes, dist_summary_p_nnodes):
    N_ranks_per_node =  parms.N_ranks_per_node
    N_images_per_rank = parms.N_images_per_rank
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_nodes = N_procs // N_ranks_per_node
    curr_index = 0
    for i in IndexLaunch([N_procs]):
        print(f"i = {i}", flush=True)
        match_task(
            phased, slices_p[i], ref_orientations_p[i], match_summary_p[i],
            dist_summary_p[i], nonuniform_batch_p[i], pixel_position, pixel_distance)

    for i in IndexLaunch([N_nodes]):
        select_orientations(match_summary_p_nnodes[i], dist_summary_p_nnodes[i], orientations_p[i])

    return orientations, orientations_p
