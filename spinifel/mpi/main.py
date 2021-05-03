from mpi4py import MPI

from spinifel import parms, utils

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase
from .orientation_matching import match
from spinifel.slicing import gen_model_slices_batch
import os

import skopi as skp
import numpy as np


def main():
    comm = MPI.COMM_WORLD

    logger = utils.Logger(comm.rank==(2 if parms.use_psana else 0))
    logger.log("In MPI main")

    N_images_per_rank = parms.N_images_per_rank


    # ADD DEBUG TEST for orientation mathcihg 
    # This will replace some input parameters
    # and exit after orientation matching module
    DEBUG_FLAG = int(os.environ.get('DEBUG_FLAG', '0'))
    if DEBUG_FLAG:
        print(f"***WARNING: DEBUG_FLAG ON***")
        print(f"  THIS WILL RESET parms.N_clipping and parms.N_binning TO 0.")
        print(f"  THEN EXIT AFTER THE FIRST ORIENTATION MATCHING.")
        parms.N_clipping = 0
        parms.N_binning = 0
        parms.N_batch_size = N_images_per_rank
        parms.reduced_det_shape = parms.det_shape
        parms.reduced_pixel_position_shape = (3,) + parms.reduced_det_shape
        parms.reduced_pixel_index_shape = (2,) + parms.reduced_det_shape


    timer = utils.Timer()

    ds = None
    if parms.use_psana:
        from psana import DataSource
        logger.log("Using psana")
        N_big_data_nodes = comm.size - 2
        batch_size = min(N_images_per_rank, 100)
        max_events = min(parms.N_images_max, N_big_data_nodes*N_images_per_rank)
        def destination(timestamp):
            # Return big data node destination, numbered from 1, round-robin
            destination.last = destination.last % N_big_data_nodes + 1
            return destination.last
        destination.last = 0
        ds = DataSource(exp=parms.exp, run=parms.runnum, dir=parms.data_dir,
                        destination=destination, max_events=max_events)

    (pixel_position_reciprocal,
     pixel_distance_reciprocal,
     pixel_index_map,
     slices_,
     orientations_,
     volume) = get_data(N_images_per_rank, ds)
    logger.log(f"Loaded in {timer.lap():.2f}s.")
    
    N_pixels = utils.prod(parms.reduced_det_shape)
    
    # flatten data
    slices_ = slices_.reshape((slices_.shape[0], N_pixels))
    
    reciprocal_extent = pixel_distance_reciprocal.max()

    
    if DEBUG_FLAG:
        print(f'WARNING: Running in debug mode - known orientations and volumes will be used', flush=True)
        ivol = np.square(np.abs(volume))
        rho_ = np.abs(np.fft.ifftn(ivol))
        support_ = None
        ac_phased = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))).astype(np.float32)
        ref_orientations = orientations_

    else:
        ac = solve_ac(
            0, pixel_position_reciprocal, pixel_distance_reciprocal, slices_)
        logger.log(f"AC recovered in {timer.lap():.2f}s. ac={ac.shape} {ac.dtype}")

        ac_phased, support_, rho_ = phase(0, ac)
        logger.log(f"Problem phased in {timer.lap():.2f}s. ac_phased={ac_phased.shape} {ac_phased.dtype}")

        # generate uniform orientations
        ref_orientations = skp.get_uniform_quat(parms.N_orientations, True)

    
    # Generate model slices using forward transform
    ac_support_size = parms.M
    model_slices = gen_model_slices_batch(ac_phased, ref_orientations, 
            pixel_position_reciprocal, reciprocal_extent, parms.oversampling, 
            ac_support_size, N_pixels, batch_size=parms.N_batch_size)
    model_slices = model_slices.reshape((ref_orientations.shape[0], N_pixels))
    logger.log(f"Slicing done in {timer.lap():.2f}s.")
    
    
    if DEBUG_FLAG:
        print(f'Input data:', flush=True)
        print(f'  slices_: {slices_.shape} dtype: {slices_.dtype}', flush=True)
        print(f'  pixel_position_reciprocal: {pixel_position_reciprocal.shape} dtype: {pixel_position_reciprocal.dtype}', flush=True)
        print(f'  pixel_distance_reciprocal: {pixel_distance_reciprocal.shape} dtype: {pixel_distance_reciprocal.dtype}', flush=True)
        print(f'  reciprocal_extent: {reciprocal_extent} dtype: {type(reciprocal_extent)}', flush=True)
        
        print(f'  ac_phased: {ac_phased.shape} dtype: {ac_phased.dtype}', flush=True)
        print(f'  model_slices: {model_slices.shape} dtype: {model_slices.dtype}', flush=True)
        print(f'  ref_orientations: {ref_orientations.shape} dtype: {ref_orientations.dtype}', flush=True)


    # scale model_slices
    data_model_scaling_ratio = slices_.std() / model_slices.std()
    logger.log(f"Data/Model std ratio: {data_model_scaling_ratio}.")
    model_slices *= data_model_scaling_ratio
    
    # Use imrovement of cc(prev_rho, cur_rho) to detemine if
    # we should terimate the loop
    cov_xy = 0 
    cov_delta = .05
    
    if DEBUG_FLAG: 
        N_generations = 3
    else:
        N_generations = parms.N_generations

    for generation in range(1, N_generations):
        logger.log(f"Generation: {generation}")
        new_orientations = match(slices_, model_slices, 
                ref_orientations, batch_size=parms.N_batch_size)
        logger.log(f"Orientations matched in {timer.lap():.2f}s.")

        if DEBUG_FLAG:
            print("COMPARE KNOWN VERSUS COMPUTED ORIENTATIONS AND DOT PRODUCT", flush=True)
            for i in range(slices_.shape[0]):
                orientations_dot_prod = np.dot(ref_orientations[i], new_orientations[i])
                print(i, ref_orientations[i], new_orientations[i],orientations_dot_prod, flush=True)
                assert orientations_dot_prod - 1.0 < 1e12

        orientations = new_orientations

        ac = solve_ac(
            generation, pixel_position_reciprocal, pixel_distance_reciprocal,
            slices_, orientations, ac_phased)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")

        if comm.rank == 0: prev_rho_ = rho_[:]
        ac_phased, support_, rho_ = phase(generation, ac, support_, rho_)
        
        if comm.rank == 0:
            cc_matrix = np.corrcoef(prev_rho_.flatten(), rho_.flatten())
            prev_cov_xy = cov_xy
            cov_xy = cc_matrix[0,1]
        else:
            prev_cov_xy = None
            cov_xy = None
        prev_cov_xy = comm.bcast(prev_cov_xy, root=0)
        cov_xy = comm.bcast(cov_xy, root=0)

        logger.log(f"Problem phased in {timer.lap():.2f}s. cc={cov_xy:.2f} delta={cov_xy-prev_cov_xy:.2f}")
        if cov_xy - prev_cov_xy < cov_delta:
            break
        
        # Calulate new model slices from new phased ac using
        # the original orientations.
        model_slices = gen_model_slices_batch(ac_phased, ref_orientations, 
                pixel_position_reciprocal, reciprocal_extent, parms.oversampling, 
                ac_support_size, N_pixels, batch_size=parms.N_batch_size)
        model_slices = model_slices.reshape((ref_orientations.shape[0], N_pixels))
        logger.log(f"Slicing done in {timer.lap():.2f}s.")


        

    logger.log(f"Total: {timer.total():.2f}s.")
