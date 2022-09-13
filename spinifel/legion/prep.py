import h5py
import numpy  as np
import PyNVTX as nvtx
import os
import pygion
import socket
from pygion import task, Tunable, Partition, Region, WD, RO, RW, Reduce, IndexLaunch

from spinifel import settings, prep, image

from . import utils as lgutils
from spinifel.extern.nufft_ext import NUFFT
from spinifel.sequential.orientation_matching import SNM
from spinifel.sequential.autocorrelation import Merge

all_objs  = {}

psana = None
if settings.use_psana:
    import psana
    from psana.psexp.legion_node import smd_chunks, smd_batches, batch_events
    from psana.psexp.legion_node import smd_batches_without_transitions, smd_chunks_steps

@task(leaf=True, privileges=[WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_pixel_position(pixel_position):
    prep.load_pixel_position_reciprocal(pixel_position.reciprocal)


@task(privileges=[WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_pixel_position_psana(pixel_position,run):
    pygion.fill(pixel_position, 'reciprocal', 0.0)
    prep.load_pixel_position_reciprocal_psana(run,
                                              pixel_position.reciprocal)

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

@task(leaf=True, privileges=[WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_pixel_index(pixel_index):
    prep.load_pixel_index_map(pixel_index.map)


@task(leaf=True, privileges=[WD])
@lgutils.gpu_task_wrapper
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
@task(leaf=True, privileges=[WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_slices_psana(slices, rank, N_images_per_rank, smd_chunk, run):
    i = 0
    for smd_batch in smd_batches(smd_chunk, run):
        for evt in batch_events(smd_batch, run):
            raw = evt._dgrams[0].pnccdBack[0].raw
            try:
                slices.data[i] = raw.image
                print(f' {rank} loading {i}')
            except IndexError:
                raise RuntimeError(
                    f"Rank {rank} received too many events.")
            i += 1
            if i==N_images_per_rank:
                print(f'Rank {rank} returning early i={i}, N_images_per_rank={N_images_per_rank}')
                return
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} loaded slices.", flush=True)


@task(leaf=True, privileges=[WD])
@lgutils.gpu_task_wrapper
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
            i = i+1
    # FIXME:
    if i < N_images_per_rank:
        for j in range(N_images_per_rank - i):
            slices.data[j+i] = slices.data[j+i-1]
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} loaded {i} slices.", flush=True)

@task(leaf=True, privileges=[WD])
@lgutils.gpu_task_wrapper
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
                pixel_position = get_pixel_position(run)
            val= val+1
            gen_smd = smd_chunks_steps(run)
            while chunk_i < n_nodes:
                for smd_chunk, step_data in gen_smd:
                    i = chunk_i % n_nodes
                    load_slices_psana2(slices_p[i], i,
                                       settings.N_images_per_rank,
                                       bytearray(smd_chunk),
                                       run,
                                       point=i)
                    chunk_i += 1
                    if chunk_i==n_nodes:
                        break
                gen_smd = smd_chunks_steps(run)
    else:
        for i, slices_subr in enumerate(slices_p):
            load_slices_hdf5(slices_subr, i, N_images_per_rank, point=i)
    return slices, slices_p, pixel_position, pixel_index


@task(leaf=True, privileges=[WD])
@lgutils.gpu_task_wrapper
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


@task(leaf=True, privileges=[RO, Reduce('+')])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def reduce_mean_image(slices, mean_image, nprocs):
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} reduce_mean_image", flush=True)
    mean_image.data[:] += slices.data.mean(axis=0) / nprocs
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} finished reduce_mean_image", flush=True)

@nvtx.annotate("legion/prep.py", is_prefix=True)
def compute_mean_image(slices, slices_p):
    mean_image = Region(lgutils.get_region_shape(slices)[1:],
                        {'data': pygion.float32})
    pygion.fill(mean_image, 'data', 0.)
    nprocs = Tunable.select(Tunable.GLOBAL_PYS).get()

    for i, sl in enumerate(slices_p):
        reduce_mean_image(sl, mean_image, nprocs, point=i)
    return mean_image



@task(leaf=True, privileges=[RO, WD])
@lgutils.gpu_task_wrapper
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



@task(leaf=True, privileges=[RO, WD])
@lgutils.gpu_task_wrapper
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



@task(leaf=True, privileges=[RO, WD])
@lgutils.gpu_task_wrapper
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



@task(leaf=True, privileges=[RO, WD])
@lgutils.gpu_task_wrapper
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
    i=0
    for old_slices_subr, new_slices_subr in zip(old_slices_p, new_slices_p):
        apply_slices_binning(old_slices_subr, new_slices_subr, point=i)
        i = i+1
    return new_slices, new_slices_p

#perform binning by copying the old data without creating a new region
@task(leaf=True, privileges=[RW])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def apply_slices_binning_new(slices):
    data[:] = slices.data
    slices.data[:] = prep.binning_sum(data)


@task(leaf=True, privileges=[RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def show_image(pixel_index, images, image_index, name):
    image.show_image(pixel_index.map, images.data[image_index], name)

@task(leaf=True, privileges=[RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def export_saxs(pixel_distance, mean_image, name):
    np.seterr(invalid='ignore')
    # Avoid warning in SAXS 0/0 division.
    # Legion seems to reset the global seterr from parms.py.
    prep.export_saxs(pixel_distance.reciprocal, mean_image.data, name)

@nvtx.annotate("legion/prep.py", is_prefix=True)
def init_partitions_regions_psana2():
    # minimum batch size = n_images_per_rank
    n_images_per_rank = settings.N_images_per_rank
    n_points = Tunable.select(Tunable.GLOBAL_PYS).get()

    # batch size
    batch_size = settings.N_images_per_rank

    # total batches per rank
    # N_images_max = maximum # of images in total
    # i.e size of the region
    assert settings.N_images_max%settings.N_images_per_rank == 0

    total_batches = settings.N_images_max//settings.N_images_per_rank

    # max batches per iteration
    max_batches = settings.N_image_batches_max

    # N_max_images_per_rank
    max_images_per_rank = settings.N_images_max

    # max batch size per iter per rank
    # settings.N_image_batches_max*batch_size
    # max_batch_size = max_batches*batch_size

    fields_dict = {"data": getattr(pygion, settings.data_type_str)}
    sec_shape = settings.reduced_det_shape

    slices  = lgutils.create_max_region(max_images_per_rank,fields_dict,sec_shape,n_points)
    # start at the first batch
    cur_batch_size = 0
    # create all the partitions
    p = lgutils.init_partitions(slices,n_points,batch_size,max_images_per_rank,cur_batch_size,sec_shape)

    # create settings.N_batches_max*batch_size det_shape slices per rank
    sec_shape = settings.det_shape
    max_images_per_iter = max_batches*batch_size
    slices_images  = lgutils.create_max_region(max_images_per_iter,fields_dict,sec_shape,n_points)
    slices_images_p = lgutils.init_partitions(slices_images,n_points,batch_size,max_images_per_iter,
                                              cur_batch_size,sec_shape)
    pygion.execution_fence(block=True)
    return slices, p, slices_images, slices_images_p

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

    for i, sl in enumerate(slices_p):
        fluctuation_task(pixel_distance,mean_image,sl,point=i)

    return (pixel_position, pixel_distance, pixel_index, slices, slices_p)


# Fluctuation
@task(leaf=True, privileges=[RO, RO, RW])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def fluctuation_task(pixel_distance, mean_image, slices_p):
    N_images_per_rank = slices_p.ispace.domain.extent[0]
    saxs_qs, mean_saxs = prep.get_saxs(pixel_distance.reciprocal, mean_image.data)
    numQ = len(saxs_qs)
    fracQ = 4 # only use fraction of the saxs curve to compute fluctuation
    intensity_thr = 0.1 # avoid dividing by small intensity
    for i in range(N_images_per_rank):
        _, single_saxs = prep.get_saxs(pixel_distance.reciprocal, slices_p.data[i])
        ind = np.where(single_saxs[:numQ//fracQ] > intensity_thr)
        factor = np.mean(mean_saxs[:numQ//fracQ][ind] / (single_saxs[:numQ//fracQ][ind]) )
        slices_p.data[i] = slices_p.data[i] * factor

#load pixel data from psana2
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_pixel_data(ds):
    pixel_position = None
    pixel_index = None
    assert ds is not None
    n_nodes = Tunable.select(Tunable.NODE_COUNT).get()
    gen_run = ds.runs()
    for run in gen_run:
        # load pixel index map and pixel position reciprocal only once
        pixel_index = get_pixel_index(run)
        pixel_position = get_pixel_position(run)
        break

    pixel_distance = compute_pixel_distance(pixel_position)
    return pixel_position, pixel_distance, pixel_index, run


#process pixel data after first set of slices are loaded
@nvtx.annotate("legion/prep.py", is_prefix=True)
def bin_slices_new(old_slices_p, new_slices_p):
    i=0
    for old_slices_subr, new_slices_subr in zip(old_slices_p, new_slices_p):
        apply_slices_binning(old_slices_subr, new_slices_subr, point=i)
        i=i+1


@nvtx.annotate("legion/prep.py", is_prefix=True)
def process_data(slices, slices_p, slices_bin, slices_bin_p,
                 pixel_distance, pixel_index, pixel_position, iteration):
    #returns a region that contains the mean of images
    mean_image = compute_mean_image(slices,slices_p)

    show_image(pixel_index, slices_p[0], 0, "image_{iteration}.png")
    show_image(pixel_index, mean_image, ..., "mean_image_{iteration}.png")
    export_saxs(pixel_distance, mean_image, "saxs_{iteration}.png")

    # bin pixel position and pixel index
    # return a region that contains the binned pixel_position + pixel_index
    if iteration == 0:
        pixel_position = bin_pixel_position(pixel_position)
        pixel_index = bin_pixel_index(pixel_index)

    # bin slices new is passed the current set of images,
    # and the slices_bin partition to copy the binned
    # slices into
    bin_slices_new(slices_p, slices_bin_p)
    # get the mean of the binned slices
    mean_image = compute_mean_image(slices_bin, slices_bin_p)
    show_image(pixel_index, slices_bin_p[0], 0, "image_binned_{iteration}.png")
    show_image(pixel_index, mean_image, ..., "mean_image_binned_{iteration}.png")

    # return a region pixel_distance based on the binned pixel_position
    # done only on iteraton 1
    if iteration == 0:
        pixel_distance = compute_pixel_distance(pixel_position)

    export_saxs(pixel_distance, mean_image, "saxs_binned_{iteration}.png")

    for i, sl in enumerate(slices_bin_p):
        fluctuation_task(pixel_distance, mean_image, sl, point=i)

    # mean_image region is no longer needed
    pygion.fill(mean_image, 'data', 0)

    # return regions containing binned pixel_position, pixel_distance, pixel_index
    return (pixel_position, pixel_distance, pixel_index)

#load image batch from psana2
@nvtx.annotate("legion/prep.py", is_prefix=True)
def load_image_batch(run, gen_run, gen_smd, slices_p):
    # create a new region of full size and load the images
    assert gen_run is not None
    n_nodes = Tunable.select(Tunable.NODE_COUNT).get()
    chunk_i = 0
    if run is None:
        run = next(gen_run)
    if gen_smd is None:
        gen_smd = smd_chunks_steps(run)
    while chunk_i < n_nodes:
        for smd_chunk, step_data in gen_smd:
            i = chunk_i % n_nodes
            load_slices_psana2(slices_p[i], i,
                               settings.N_images_per_rank,
                               bytearray(smd_chunk),
                               run,
                               point=i)
            chunk_i += 1
            if chunk_i==n_nodes:
                break
        gen_smd = smd_chunks_steps(run)
    return gen_run, gen_smd, run


@task(leaf=True, privileges=[RO, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/prep.py", is_prefix=True)
def setup_objects_task(pixel_position, pixel_distance, slices):
    global all_objs
    N_images_per_rank = slices.ispace.domain.extent[0]
    all_objs['nufft'] = NUFFT(settings, pixel_position.reciprocal,
                              pixel_distance.reciprocal, N_images_per_rank)
    all_objs['snm'] = SNM(
        settings,
        slices.data,
        pixel_position.reciprocal,
        pixel_distance.reciprocal,
        all_objs['nufft'])

    all_objs['mg'] = Merge(
        settings,
        slices.data,
        pixel_position.reciprocal,
        pixel_distance.reciprocal,
        all_objs['nufft'])
    done = True
    return done

@nvtx.annotate("legion/prep.py", is_prefix=True)
def prep_objects(pixel_position, pixel_distance, slices, N_procs):
    done_list = []
    for i in range(N_procs):
        done = setup_objects_task(pixel_position, pixel_distance, slices[i], point=i)
        done_list.append(done)
    return done_list
