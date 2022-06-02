from spinifel import settings, utils, contexts, checkpoint, image
from spinifel.prep import save_mrc, compute_pixel_distance, binning_mean, binning_index 

import numpy as np
import PyNVTX as nvtx

from .prep import get_data, compute_mean_image, show_image, bin_data
from .autocorrelation import solve_ac
from .phasing import phase
from .orientation_matching import match



@nvtx.annotate("mpi/main.py", is_prefix=True)
def main():
    print("Enter the dragon",flush=True)
    comm = contexts.comm

    print(f"#### comm: {comm}",flush=True)
    #print(f"#### comm_group: {comm.Get_group()}",flush=True)
    #print(f"#### size: {comm.size}",flush=True)
    #print(f"#### rank: {comm.rank}",flush=True)

    timer = utils.Timer()

    # Reading input images from hdf5
    N_images_per_rank = settings.N_images_per_rank
    batch_size = min(N_images_per_rank, 100)
    N_big_data_nodes = comm.size
    max_events = min(settings.N_images_max, N_big_data_nodes*N_images_per_rank)
    writer_rank = 0 # pick writer rank as core 0
    print(f"Got here",flush=True)

    # Reading input images using psana2
    ds = None
    if settings.use_psana:
        print(f"load data from psana",flush=True)
        from psana import DataSource
        # BigData cores are those excluding Smd0, EventBuilder, & Server cores.
        N_big_data_nodes = comm.size - (1 + settings.ps_eb_nodes + settings.ps_srv_nodes)
        writer_rank = 1 + settings.ps_eb_nodes # pick writer rank as the first BigData core

        # Limit batch size to 100
        batch_size = min(N_images_per_rank, 100)

        # Calculate total no. of images that will be processed (limit by max)
        max_events = min(settings.N_images_max, N_big_data_nodes*N_images_per_rank)
        
        def destination(timestamp):
            # Return big data node destination, numbered from 1, round-robin
            destination.last = destination.last % N_big_data_nodes + 1
            return destination.last
        destination.last = 0

        # Create a datasource and ask for images. For example, 
        # batch_size = 100, N_images_per_rank = 4000, N_big_data_nodes = 3
        # -- > max_events = 12000
        # The destination callback above sends events to BigData cores
        # in round robin order.
        print(f"exp:{settings.ps_exp, settings.ps_runnum, settings.ps_dir, destination}",flush=True)
        ds = DataSource(exp=settings.ps_exp, run=settings.ps_runnum,
                        dir=settings.ps_dir, destination=destination,
                        max_events=max_events)
    print("Done loading",flush=True)

    # Setup logger after knowing the writer rank 
    logger = utils.Logger(comm.rank==writer_rank)
    logger.log("In MPI main")
    if settings.use_psana:
        logger.log("Using psana")
    logger.log(f"comm.size : {comm.size:d}")
    logger.log(f"#workers  : {N_big_data_nodes:d}")
    logger.log(f"writerrank: {writer_rank}")
    logger.log(f"batch_size: {batch_size}")
    logger.log(f"max_events: {max_events}")

    # Load unique set of intensity slices for each rank
    # In psana2 mode, get_data loops over the event loop 
    # until the data array is filled with N_images_per_rank
    # events.
    (pixel_position_reciprocal,
     pixel_index_map,
     slices_) = get_data(N_images_per_rank, ds)
    
    # Hacky way to allow only worker ranks for computation tasks
    if not contexts.is_worker: return
    
    # Computes reciprocal distance and mean image then save to .png
    pixel_distance_reciprocal = compute_pixel_distance(
            pixel_position_reciprocal)
    mean_image = compute_mean_image(slices_)
    show_image(image, ds, contexts.rank, slices_, pixel_index_map, 
            pixel_position_reciprocal, pixel_distance_reciprocal, mean_image,
            "image_0.png", "mean_image.png", "saxs.png")

    # Bins data and save to .png files
    (pixel_position_reciprocal,
     pixel_index_map,
     slices_) = bin_data(pixel_position_reciprocal, pixel_index_map, slices_)
    pixel_distance_reciprocal = compute_pixel_distance(
            pixel_position_reciprocal)
    mean_image = compute_mean_image(slices_)
    show_image(image, ds, contexts.rank, slices_, pixel_index_map, 
            pixel_position_reciprocal, pixel_distance_reciprocal, mean_image,
            "image_binned_0.png", "mean_image_binned.png", "saxs_binned.png")
    
    logger.log(f"Loaded in {timer.lap():.2f}s.")
    

    # Generation 0: solve_ac and phase
    N_generations = settings.N_generations

    if settings.load_gen > 0: # Load input from previous generation
        curr_gen = settings.load_gen
        print(f"Loading checkpoint: {checkpoint.generate_checkpoint_name(settings.out_dir, settings.load_gen, settings.tag_gen)}", flush=True)
        myRes = checkpoint.load_checkpoint(settings.out_dir, 
                                           settings.load_gen, 
                                           settings.tag_gen)
        # Unpack dictionary
        ac_phased = myRes['ac_phased']
        support_ = myRes['support_']
        rho_ = myRes['rho_']
        orientations = myRes['orientations']
    else:
        curr_gen = 0
        logger.log(f"#"*27)
        logger.log(f"##### Generation {curr_gen}/{N_generations} #####")
        logger.log(f"#"*27)
        ac = solve_ac(
            curr_gen, pixel_position_reciprocal, pixel_distance_reciprocal, slices_)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")
        if comm.rank == 0:
            myRes = { 
                     'pixel_position_reciprocal': pixel_position_reciprocal,
                     'pixel_distance_reciprocal': pixel_distance_reciprocal,
                     'slices_': slices_,
                     'ac': ac
                    }
            checkpoint.save_checkpoint(myRes, settings.out_dir, curr_gen, tag="solve_ac")

        ac_phased, support_, rho_ = phase(curr_gen, ac)
        logger.log(f"Problem phased in {timer.lap():.2f}s.")
        if comm.rank == 0:
            myRes = { 
                     'ac': ac,
                     'ac_phased': ac_phased,
                     'support_': support_,
                     'rho_': rho_
                    }
            checkpoint.save_checkpoint(myRes, settings.out_dir, curr_gen, tag="phase")
            # Save electron density and intensity
            rho = np.fft.ifftshift(rho_)
            intensity = np.fft.ifftshift(np.abs(np.fft.fftshift(ac_phased)**2))
            save_mrc(settings.out_dir / f"ac-{curr_gen}.mrc", ac_phased)
            save_mrc(settings.out_dir / f"intensity-{curr_gen}.mrc", intensity)
            save_mrc(settings.out_dir / f"rho-{curr_gen}.mrc", rho)

    # Use improvement of cc(prev_rho, cur_rho) to dertemine if we should
    # terminate the loop
    cov_xy = 0
    cov_delta = .05
    curr_gen += 1

    for generation in range(curr_gen, N_generations+1):
        logger.log(f"#"*27)
        logger.log(f"##### Generation {generation}/{N_generations} #####")
        logger.log(f"#"*27)
        # Orientation matching
        orientations = match(
            ac_phased, slices_,
            pixel_position_reciprocal, pixel_distance_reciprocal)
        logger.log(f"Orientations matched in {timer.lap():.2f}s.")
        if comm.rank == 0:
            myRes = {'ac_phased': ac_phased, 
                     'slices_': slices_,
                     'pixel_position_reciprocal': pixel_position_reciprocal,
                     'pixel_distance_reciprocal': pixel_distance_reciprocal,
                     'orientations': orientations
                    }
            checkpoint.save_checkpoint(myRes, settings.out_dir, generation, tag="match")

        # Solve autocorrelation
        ac = solve_ac(
            generation, pixel_position_reciprocal, pixel_distance_reciprocal,
            slices_, orientations, ac_phased)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")
        if comm.rank == 0:
            myRes = { 
                     'pixel_position_reciprocal': pixel_position_reciprocal,
                     'pixel_distance_reciprocal': pixel_distance_reciprocal,
                     'slices_': slices_,
                     'orientations': orientations,
                     'ac_phased': ac_phased,
                     'ac': ac
                    }
            checkpoint.save_checkpoint(myRes, settings.out_dir, generation, tag="solve_ac")

        if comm.rank == 0: 
            prev_rho_ = rho_[:]
            prev_support_ = support_[:]
        ac_phased, support_, rho_ = phase(generation, ac, support_, rho_)
        logger.log(f"Problem phased in {timer.lap():.2f}s.")
        if comm.rank == 0:
            myRes = { 
                     'ac': ac,
                     'prev_support_':prev_support_,
                     'prev_rho_': prev_rho_,
                     'ac_phased': ac_phased,
                     'support_': support_,
                     'rho_': rho_
                    }
            checkpoint.save_checkpoint(myRes, settings.out_dir, generation, tag="phase")


        # Check if density converges
        if settings.chk_convergence:
            # Calculate correlation coefficient
            if comm.rank == 0:
                prev_cov_xy = cov_xy
                cov_xy = np.corrcoef(prev_rho_.flatten(), rho_.flatten())[0,1]
            else:
                prev_cov_xy = None
                cov_xy = None
            logger.log(f"CC in {timer.lap():.2f}s. cc={cov_xy:.2f} delta={cov_xy-prev_cov_xy:.2f}")
            
            # Stop if improvement in cc is less than cov_delta
            prev_cov_xy = comm.bcast(prev_cov_xy, root=0)
            cov_xy = comm.bcast(cov_xy, root=0)
            if cov_xy - prev_cov_xy < cov_delta:
                print("Stopping criteria met!")
                break

        if comm.rank == 0:
            # Save electron density and intensity
            rho = np.fft.ifftshift(rho_)
            intensity = np.fft.ifftshift(np.abs(np.fft.fftshift(ac_phased)**2))
            save_mrc(settings.out_dir / f"ac-{generation}.mrc", ac_phased)
            save_mrc(settings.out_dir / f"intensity-{generation}.mrc", intensity)
            save_mrc(settings.out_dir / f"rho-{generation}.mrc", rho)

            # Save output
            myRes = {'ac_phased': ac_phased, 
                     'support_': support_,
                     'rho_': rho_,
                     'orientations': orientations
                    }
            checkpoint.save_checkpoint(myRes, settings.out_dir, generation)

    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")
