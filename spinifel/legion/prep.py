import h5py
import numpy  as np
import PyNVTX as nvtx
import os
import pygion
import socket
from pygion import task, Tunable, Partition, Region, WD, RO, Reduce, IndexLaunch

from spinifel import settings, prep, image

from . import utils as lgutils



psana = None
if settings.use_psana:
    import psana
    from psana.psexp.legion_node import smd_chunks, smd_batches, batch_events
    from psana.psexp.legion_node import smd_batches_without_transitions, smd_chunks_steps

@task(privileges=[WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_pixel_position(pixel_position):
    prep.load_pixel_position_reciprocal(pixel_position.reciprocal)


@task(privileges=[WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_pixel_position_psana(pixel_position,run):
    pixel_position_val = 0
    if run.expt == "xpptut15":
        prep.load_pixel_position_reciprocal_psana(run,
                                                  pixel_position_val,
                                                  pixel_position.reciprocal)
    # TODO - this needs to be setup for psana
    elif run.expt == "amo06516":
        pygion.fill(pixel_position, 'reciprocal', 0)
    else:
        assert False

@nvtx.annotate("legion/prep.py", is_prefix=True)
def get_pixel_position(run=None):
    pixel_position_type = getattr(pygion, settings.pixel_position_type_str)
    pixel_position = Region(settings.pixel_position_shape,
                            {'reciprocal': pixel_position_type})
    if run == None:
        load_pixel_position(pixel_position)
    else:
        load_pixel_position_psana(pixel_position, run)
    return pixel_position

@task(privileges=[WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_pixel_index(pixel_index):
    prep.load_pixel_index_map(pixel_index.map)


@task(privileges=[WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_pixel_index_psana(pixel_index, run):
    pixel_index.map[:] = np.moveaxis(run.beginruns[0].scan[0].raw.pixel_index_map, -1, 0)

@nvtx.annotate("legion/prep.py", is_prefix=True)
def get_pixel_index(run=None):
    pixel_index_type = getattr(pygion, settings.pixel_index_type_str)
    pixel_index = Region(settings.pixel_index_shape,
                         {'map': pixel_index_type})
    # psana
    if run:
        load_pixel_index_psana(pixel_index, run)
    else:
        load_pixel_index(pixel_index)
    return pixel_index


# this is the equivalent of big data (the loop around batch_events)
@task(privileges=[WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_slices_psana(slices, rank, N_images_per_rank, smd_chunk, run):
    i = 0
    for smd_batch in smd_batches(smd_chunk, run):
        for evt in batch_events(smd_batch, run):
            raw = evt._dgrams[0].pnccdBack[0].raw
            try:
                slices.data[i] = raw.image
            except IndexError:
                raise RuntimeError(
                    f"Rank {rank} received too many events.")
            i += 1
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} loaded slices.", flush=True)


@task(privileges=[WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_slices_psana2(slices, rank, N_images_per_rank, smd_chunk, run):
    i = 0
    if settings.verbosity > 0:
        print(f'{socket.gethostname()} load_slices_psana2, rank = {rank}',flush=True)
    det = run.Detector("amopnccd")
    for smd_batch in smd_batches_without_transitions(smd_chunk, run):
        for evt in batch_events(smd_batch,run):
            raw = det.raw.calib(evt)
            try:
                slices.data[i] = raw
            except IndexError:
                raise RuntimeError(
                    f"Task received too many events.")
            i += 1
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} loaded slices.", flush=True)

@task(privileges=[WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_slices_hdf5(slices, rank, N_images_per_rank):
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} loading slices.", flush=True)
    i_start = rank * N_images_per_rank
    i_end = i_start + N_images_per_rank
    prep.load_slices(slices.data, i_start, i_end)
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} loaded slices.", flush=True)

@nvtx.annotate("legion/prep.py", is_prefix=True)
def get_slices(ds):
    N_images_per_rank = settings.N_images_per_rank
    fields_dict = {"data": getattr(pygion, settings.data_type_str)}
    sec_shape = settings.det_shape
    slices, slices_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    pixel_position = None
    pixel_index = None
    if ds is not None:
        n_nodes = Tunable.select(Tunable.NODE_COUNT).get()
        chunk_i = 0
        val = 0
        for run in ds.runs():
            # load pixel index map and pixel position reciprocal only once
            if val == 0:
                # pixel index map
                pixel_index = get_pixel_index(run)
                # TODO pixel_position.reciprocal for "am06516"
                pixel_position = get_pixel_position(run)
            val= val+1
            for smd_chunk, step_data in smd_chunks_steps(run):
                i = chunk_i % n_nodes
                load_slices_psana2(slices_p[i], i, N_images_per_rank, bytearray(smd_chunk), run, point=i)
                chunk_i += 1
        pygion.execution_fence(block=True)
    else:
        for i, slices_subr in enumerate(slices_p):
            load_slices_hdf5(slices_subr, i, N_images_per_rank, point=i)
    return slices, slices_p, pixel_position, pixel_index


@task(privileges=[WD])
def load_orientations_prior(orientations_prior, rank, N_images_per_rank):
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} loading orientations.", flush=True)
    i_start = rank * N_images_per_rank
    i_end = i_start + N_images_per_rank
    prep.load_orientations_prior(orientations_prior.quaternions, i_start, i_end)
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} loaded orientations.", flush=True)


def get_orientations_prior():
    N_images_per_rank = settings.N_images_per_rank
    fields_dict = {"quaternions": getattr(pygion, settings.data_type_str)}
    sec_shape = settings.quaternion_shape
    orientations_prior, orientations_prior_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    for i, orientations_prior_subr in enumerate(orientations_prior_p):
        load_orientations_prior(orientations_prior_subr, i, N_images_per_rank, point=i)
    return orientations_prior, orientations_prior_p


@task(privileges=[RO, Reduce('+')])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def reduce_mean_image(slices, mean_image):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    mean_image.data[:] += slices.data.mean(axis=0) / N_procs



@nvtx.annotate("legion/prep.py", is_prefix=True)
def compute_mean_image(slices, slices_p):
    mean_image = Region(lgutils.get_region_shape(slices)[1:],
                        {'data': pygion.float32})
    pygion.fill(mean_image, 'data', 0.)
    for slices in slices_p:
        reduce_mean_image(slices, mean_image)
    return mean_image



@task(privileges=[RO, WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def calculate_pixel_distance(pixel_position, pixel_distance):
    pixel_distance.reciprocal[:] = prep.compute_pixel_distance(
            pixel_position.reciprocal)



@nvtx.annotate("legion/prep.py", is_prefix=True)
def compute_pixel_distance(pixel_position):
    pixel_position_type = getattr(pygion, settings.pixel_position_type_str)
    pixel_distance = Region(lgutils.get_region_shape(pixel_position)[1:],
                            {'reciprocal': pixel_position_type})
    calculate_pixel_distance(pixel_position, pixel_distance)
    return pixel_distance



@task(privileges=[RO, WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def apply_pixel_position_binning(old_pixel_position, new_pixel_position):
    new_pixel_position.reciprocal[:] = prep.binning_mean(
        old_pixel_position.reciprocal)



@nvtx.annotate("legion/prep.py", is_prefix=True)
def bin_pixel_position(old_pixel_position):
    pixel_position_type = getattr(pygion, settings.pixel_position_type_str)
    new_pixel_position = Region(settings.reduced_pixel_position_shape,
                                {'reciprocal': pixel_position_type})
    apply_pixel_position_binning(old_pixel_position, new_pixel_position)
    return new_pixel_position



@task(privileges=[RO, WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def apply_pixel_index_binning(old_pixel_index, new_pixel_index):
    new_pixel_index.map[:] = prep.binning_index(
        old_pixel_index.map)



@nvtx.annotate("legion/prep.py", is_prefix=True)
def bin_pixel_index(old_pixel_index):
    pixel_index_type = getattr(pygion, settings.pixel_index_type_str)
    new_pixel_index = Region(settings.reduced_pixel_index_shape,
                             {'map': pixel_index_type})
    apply_pixel_index_binning(old_pixel_index, new_pixel_index)
    return new_pixel_index



@task(privileges=[RO, WD])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def apply_slices_binning(old_slices, new_slices):
    new_slices.data[:] = prep.binning_sum(old_slices.data)



@nvtx.annotate("legion/prep.py", is_prefix=True)
def bin_slices(old_slices, old_slices_p):
    N_images_per_rank = settings.N_images_per_rank
    fields_dict = {"data": getattr(pygion, settings.data_type_str)}
    sec_shape = settings.reduced_det_shape
    new_slices, new_slices_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    for old_slices_subr, new_slices_subr in zip(old_slices_p, new_slices_p):
        apply_slices_binning(old_slices_subr, new_slices_subr)
    return new_slices, new_slices_p



@task(privileges=[RO, RO])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def show_image(pixel_index, images, image_index, name):
    image.show_image(pixel_index.map, images.data[image_index], name)


@task(privileges=[RO, RO])
@nvtx.annotate("legion/prep.py", is_prefix=True)
def export_saxs(pixel_distance, mean_image, name):
    np.seterr(invalid='ignore')
    # Avoid warning in SAXS 0/0 division.
    # Legion seems to reset the global seterr from parms.py.
    prep.export_saxs(pixel_distance.reciprocal, mean_image.data, name)



@nvtx.annotate("legion/prep.py", is_prefix=True)
def get_data(ds):
    print(f"{socket.gethostname()} loading slices.", flush=True)

    # if psana load pixel_position/pixel_index using first 'run'
    slices, slices_p, pixel_position, pixel_index = get_slices(ds)

    # if not psana - load pixel_postion/pixel_index from hdf5 file
    if ds == None:
        pixel_position = get_pixel_position(ds)
        pixel_index = get_pixel_index()

    mean_image = compute_mean_image(slices, slices_p)
    show_image(pixel_index, slices_p[0], 0, "image_0.png")
    show_image(pixel_index, mean_image, ..., "mean_image.png")
    pixel_distance = compute_pixel_distance(pixel_position)
    export_saxs(pixel_distance, mean_image, "saxs.png")
    pixel_position = bin_pixel_position(pixel_position)
    pixel_index = bin_pixel_index(pixel_index)
    slices, slices_p = bin_slices(slices, slices_p)
    mean_image = compute_mean_image(slices, slices_p)
    show_image(pixel_index, slices_p[0], 0, "image_binned_0.png")
    show_image(pixel_index, mean_image, ..., "mean_image_binned.png")
    pixel_distance = compute_pixel_distance(pixel_position)
    export_saxs(pixel_distance, mean_image, "saxs_binned.png")
    return (pixel_position, pixel_distance, pixel_index, slices, slices_p)
