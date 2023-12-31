from spinifel import settings, utils, contexts, checkpoint, image
from spinifel.prep import save_mrc, compute_pixel_distance, binning_mean, binning_index, load_pixel_position_reciprocal_psana

import numpy as np
import PyNVTX as nvtx
import os

from .prep import compute_mean_image, show_image, bin_data
from .phasing import phase

from spinifel.sequential.orientation_matching import SNM
from .autocorrelation import MergeMPI
from spinifel.extern.nufft_ext import NUFFT

from eval.fsc import compute_fsc, compute_reference
from eval.align import align_volumes

# Old solve_act and match for debugging in psana2 branch
from .work_autocorrelation import solve_ac as work_solve_ac
from .work_orientation_matching import match as work_match

# For main and unit tests
from .test_util import get_known_orientations

# For making sure that gpu memory is released.
import gc

if settings.use_cuda:
    import pycuda.driver as cuda
    import cupy
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()

def log_cuda_mem_info(logger):
    if settings.use_cuda:
        (free,total)=cuda.mem_get_info()
        logger.log(f"Global memory occupancy: {free*100/total:.2f}% free ({free/1e9:.2f}/{total/1e9:.2f} GB)")
        mempool_used = mempool.used_bytes()*1e-9
        mempool_total= mempool.total_bytes()*1e-9
        logger.log(f"|-->Cupy: {mempool_used=:.2f}GB {mempool_total=:.2f}GB {pinned_mempool.n_free_blocks()=:d}")

@nvtx.annotate("mpi/main.py", is_prefix=True)
def main():
    assert settings.use_psana
    
    comm = contexts.comm

    timer = utils.Timer()

    N_images_per_rank = settings.N_images_per_rank
    N_images_max = settings.N_images_max
    assert N_images_max % N_images_per_rank == 0, "N_images_max must be divisible by N_images_per_rank" 

    N_generations = settings.N_generations
    
    # BigData cores are those excluding Smd0, EventBuilder, & Server cores.
    N_big_data_nodes = comm.size - (1 + settings.ps_eb_nodes + settings.ps_srv_nodes)
    # Writer rank is the first bigdata core 
    writer_rank = 1 + settings.ps_eb_nodes  

    # Reading input images using psana2
    from psana import DataSource
    
    ds = DataSource(exp=settings.ps_exp, run=settings.ps_runnum, dir=settings.ps_dir, batch_size=settings.ps_batch_size)

    # Setup logger for all worker ranks
    logger = utils.Logger(contexts.is_worker, myrank=comm.rank)
    logger.log("In MPI main")
    if settings.use_psana:
        logger.log("Using psana")
    logger.log(f"comm.size : {comm.size:d}")
    logger.log(f"#workers  : {N_big_data_nodes:d}")
    logger.log(f"writerrank: {writer_rank}")
    logger.log(f"#img/rank : {N_images_per_rank}")
    logger.log(f"PS_SMD_N_EVENTS: {os.environ.get('PS_SMD_N_EVENTS','10000')}")
    logger.log(f"ps batch_size: {settings.ps_batch_size}")

    # Skip this data saving and ac calculation in test mode
    generation = 0
    reference_dict = {"reference": None, "dist_recip_max": None}
    if settings.load_gen > 0:  # Load input from previous generation
        generation = settings.load_gen
        print(
            f"Loading checkpoint: {checkpoint.generate_checkpoint_name(settings.out_dir, settings.load_gen, settings.tag_gen)}",
            flush=True,
        )
        myRes = checkpoint.load_checkpoint(
            settings.out_dir, settings.load_gen, settings.tag_gen
        )
        # Unpack dictionary
        ac_phased = myRes["ac_phased"]
        support_ = myRes["support_"]
        rho_ = myRes["rho_"]
        orientations = myRes["orientations"]
        generation += 1

    # Convergence check uses reference model (known answer) and compare with
    # phased model at the end of the generation. The algorithm is decided 'converged'
    # when the correlation between the known and the calculated models are above
    # min_cc and that the change from previous generation is less than min_change_cc.
    min_cc, min_change_cc = settings.fsc_min_cc, settings.fsc_min_change_cc
    final_cc, delta_cc = 0.0, 1.0
    resolution = 0.0

    # Obtain run-related info.
    run = next(ds.runs())
    det = run.Detector("amopnccd")

    _pixel_index_map = run.beginruns[0].scan[0].raw.pixel_index_map
    pixel_index_map = np.moveaxis(_pixel_index_map[:], -1, 0)
    raw_pixel_index_map = pixel_index_map

    pixel_position_reciprocal = np.zeros((3,) + settings.reduced_det_shape)
    if hasattr(run.beginruns[0].scan[0].raw, "pixel_position_reciprocal"):
        load_pixel_position_reciprocal_psana(run, pixel_position_reciprocal)
        if settings.use_single_prec:
            pixel_position_reciprocal = pixel_position_reciprocal.astype(np.float32)
        raw_pixel_position_reciprocal = pixel_position_reciprocal


    pixel_position = None
    if hasattr(run.beginruns[0].scan[0].raw, "pixel_position"):
        pixel_position = run.beginruns[0].scan[0].raw.pixel_position

    # Allocate image array to max no. images specified
    data_type = getattr(np, settings.data_type_str)
    all_slices_ = np.zeros(
        (N_images_max,) + settings.reduced_det_shape, dtype=data_type
    )

    # Allocate raw image array to N_images_per_rank (so they
    # will get replaced with new set of images)
    raw_slices_ = np.zeros((N_images_per_rank,) + settings.det_shape, dtype=data_type)

    # This flag allows us to remember that we already binned pixel data
    flag_pixel_data_binned = False

    # We need to remember where is our last slice (e.g. skip binning) processed.
    last_seen_slice = -1

    # We also count no. of processed and new images separately from i_evt
    cn_processed_events = 0
    cn_new_events = 0
    
    # For checking if we need to reinitialize nuftt et al.
    nufft = None
    log_cuda_mem_info(logger)

    flag_converged = False

    logger.log(f"Initialized in {timer.lap():.2f}s.")
    # Looping over events and run spinifel when receive enough events
    for i_evt, evt in enumerate(run.events()):
        # Quit reading when max generations reached
        if generation == N_generations: ds.terminate()

        # Only need to do once for data that needs to convert pixel_position
        if pixel_position_reciprocal is None:
            photon_energy = det.raw.photon_energy(evt)

            # Calculate pixel position in reciprocal space
            from skopi.beam import convert
            from skopi.geometry import get_reciprocal_space_pixel_position

            wavelength = convert.photon_energy_to_wavelength(photon_energy)
            wavevector = np.array([0, 0, 1.0 / wavelength])  # skopi convention
            _pixel_position_reciprocal = get_reciprocal_space_pixel_position(
                pixel_position, wavevector
            )
            pixel_position_reciprocal = np.moveaxis(
                _pixel_position_reciprocal[:], -1, 0
            )
            # Keeps a copy prior to binning
            raw_pixel_position_reciprocal = pixel_position_reciprocal

        # Start collecting slices only until max and count no. of processed
        # images. This no. is no longer increased when i_evt exceeds N_images_max.
        if i_evt < N_images_max:
            raw_slices_[cn_new_events] = det.raw.calib(evt)
            cn_new_events += 1
            if i_evt and (i_evt + 1) % N_images_per_rank == 0:
                cn_processed_events = i_evt + 1
                cn_new_events = 0
        else:
            # Set the processed event counter back to the last seen to allow
            # Spinifel to continue with the same amount of events when
            # max no. of generation is not reached
            cn_processed_events = last_seen_slice + 1

        # Call spinifel methods when we have a new set of images or the same
        # set of images (N_images_max reached) but N_generations not reached.
        if (
            cn_processed_events
            and cn_processed_events % N_images_per_rank == 0
            and generation < N_generations
        ):
            logger.log(f"#" * 42)
            logger.log(
                f"##### Generation {generation}/{N_generations} Slices:{cn_processed_events}/{N_images_max}#####"
            )
            logger.log(f"#" * 42)
            logger.log(f"Loaded in {timer.lap():.2f}s.")
            
            # Computes reciprocal distance and mean of new images then save to .png
            # prior to binning.
            raw_pixel_distance_reciprocal = compute_pixel_distance(
                raw_pixel_position_reciprocal
            )
            
            mean_image = compute_mean_image(raw_slices_)
            
            # This is only done on first worker rank - other ranks will see None for mean_image
            if mean_image is not None:
                show_image(
                    image,
                    ds,
                    contexts.rank,
                    raw_slices_,
                    raw_pixel_index_map,
                    raw_pixel_position_reciprocal,
                    raw_pixel_distance_reciprocal,
                    mean_image,
                    f"image_{generation}.png",
                    f"mean_image_{generation}.png",
                    f"saxs_{generation}.png",
                )

            # Bin pixel data (only need once)
            if not flag_pixel_data_binned:
                pixel_position_reciprocal, pixel_index_map, _ = bin_data(
                    pixel_position_reciprocal=pixel_position_reciprocal,
                    pixel_index_map=pixel_index_map,
                )
                pixel_distance_reciprocal = compute_pixel_distance(
                    pixel_position_reciprocal
                )
                flag_pixel_data_binned = True

            # Bin image data (if there are new data) and store them in permanent array
            st_slice_index = last_seen_slice + 1
            if cn_processed_events - st_slice_index > 0:
                _, _, all_slices_[st_slice_index:cn_processed_events, :] = bin_data(
                    slices_=raw_slices_
                )
                # Create a new-images-only window for binning
                new_slices_ = all_slices_[st_slice_index:cn_processed_events, :]

                # Computes reciprocal distance and mean of new images then save to .png
                # after binning.
                mean_image = compute_mean_image(new_slices_)
                if mean_image is not None:
                    show_image(
                        image,
                        ds,
                        contexts.rank,
                        new_slices_,
                        pixel_index_map,
                        pixel_position_reciprocal,
                        pixel_distance_reciprocal,
                        mean_image,
                        f"image_binned_{generation}.png",
                        f"mean_image_binned_{generation}.png",
                        f"saxs_binned_{generation}.png",
                    )

            # Create an operating window into all the slices
            slices_ = all_slices_[:cn_processed_events, :]

            logger.log(f"Images prepared in {timer.lap():.2f}s.")

            # Intitilize merge and orientation matching 
            if nufft is None:
                nufft = NUFFT(
                    settings, pixel_position_reciprocal, pixel_distance_reciprocal, cn_processed_events
                )
                mg = MergeMPI(
                    settings,
                    slices_,
                    pixel_position_reciprocal,
                    pixel_distance_reciprocal,
                    nufft,
                )

                snm = SNM(
                    settings,
                    slices_,
                    pixel_position_reciprocal,
                    pixel_distance_reciprocal,
                    nufft,
                )
                log_cuda_mem_info(logger)
            logger.log(f"Initialized NUFFT in {timer.lap():.2f}s.")

            # Solve autocorrelation first for generation 0
            if generation == 0:
                # ac = work_solve_ac(
                #    generation, pixel_position_reciprocal, pixel_distance_reciprocal, slices_)
                ac = mg.solve_ac(generation)
                logger.log(f"AC recovered in {timer.lap():.2f}s.")

                # If the pdb file is given, the writer rank will calculate this
                if settings.pdb_path.is_file() and settings.chk_convergence and comm.rank == writer_rank:
                    dist_recip_max = np.max(pixel_distance_reciprocal)
                    reference = compute_reference(
                        settings.pdb_path, settings.M, dist_recip_max
                    )
                    logger.log(f"Reference created in {timer.lap():.2f}s.")
                    reference_dict["reference"] = reference
                    reference_dict["dist_recip_max"] = dist_recip_max

                # If the checkpoint is set, the writer rank will calculate this
                if settings.checkpoint and comm.rank == writer_rank:
                    myRes = {
                        "pixel_position_reciprocal": pixel_position_reciprocal,
                        "pixel_distance_reciprocal": pixel_distance_reciprocal,
                        "slices_": slices_,
                        "ac": ac,
                    }
                    checkpoint.save_checkpoint(
                        myRes, settings.out_dir, generation, tag="solve_ac", protocol=4
                    )

                ac_phased, support_, rho_ = phase(generation, ac)
                logger.log(f"Problem phased in {timer.lap():.2f}s.")

                if settings.checkpoint and comm.rank == writer_rank:
                    myRes = {
                        **myRes,
                        **{
                            "ac": ac,
                            "ac_phased": ac_phased,
                            "support_": support_,
                            "rho_": rho_,
                        },
                    }
                    checkpoint.save_checkpoint(
                        myRes, settings.out_dir, generation, tag="phase", protocol=4
                    )
                
                # Save electron density and intensity
                if comm.rank == writer_rank:
                    rho = np.fft.ifftshift(rho_)
                    intensity = np.fft.ifftshift(
                        np.abs(np.fft.fftshift(ac_phased) ** 2)
                    )
                    save_mrc(settings.out_dir / f"ac-{generation}.mrc", ac_phased)
                    save_mrc(
                        settings.out_dir / f"intensity-{generation}.mrc", intensity
                    )
                    save_mrc(settings.out_dir / f"rho-{generation}.mrc", rho)

            # Orientation matching
            orientations = snm.slicing_and_match(ac_phased)
            # orientations = work_match(
            #    ac_phased, slices_,
            #    pixel_position_reciprocal,
            #    pixel_distance_reciprocal)
            
            # In test mode, we supply some correct orientations to guarantee convergence
            if int(os.environ.get("SPINIFEL_TEST_FLAG", "0")) and generation==0:
                logger.log(f"****WARNING**** In Test Mode - supplying {settings.fsc_fraction_known_orientations*100:.1f}% correct orientations")
                N_supply = int(settings.fsc_fraction_known_orientations * orientations.shape[0])
                orientations[:N_supply] = get_known_orientations()[:N_supply]

            logger.log(f"Orientations matched in {timer.lap():.2f}s.")

            if settings.checkpoint and comm.rank == writer_rank:
                myRes = {
                    **myRes,
                    **{
                        "ac_phased": ac_phased,
                        "slices_": slices_,
                        "pixel_position_reciprocal": pixel_position_reciprocal,
                        "pixel_distance_reciprocal": pixel_distance_reciprocal,
                        "orientations": orientations,
                    },
                }

                checkpoint.save_checkpoint(
                    myRes, settings.out_dir, generation, tag="match", protocol=4
                )

            # Solve autocorrelation
            # ac = work_solve_ac(
            #    generation, pixel_position_reciprocal, pixel_distance_reciprocal,
            #    slices_, orientations, ac_phased)
            ac = mg.solve_ac(generation, orientations, ac_phased)
            logger.log(f"AC recovered in {timer.lap():.2f}s.")

            if settings.checkpoint and comm.rank == writer_rank:
                myRes = {
                    **myRes,
                    **{
                        "pixel_position_reciprocal": pixel_position_reciprocal,
                        "pixel_distance_reciprocal": pixel_distance_reciprocal,
                        "slices_": slices_,
                        "orientations": orientations,
                        "ac_phased": ac_phased,
                        "ac": ac,
                    },
                }
                checkpoint.save_checkpoint(
                    myRes, settings.out_dir, generation, tag="solve_ac", protocol=4
                )

            ac_phased, support_, rho_ = phase(generation, ac, support_, rho_)

            logger.log(f"Problem phased in {timer.lap():.2f}s.")

            if settings.checkpoint and comm.rank == writer_rank:
                myRes = {
                    **myRes,
                    **{
                        "ac": ac,
                        "ac_phased": ac_phased,
                        "support_": support_,
                        "rho_": rho_,
                    },
                }
                checkpoint.save_checkpoint(
                    myRes, settings.out_dir, generation, tag="phase", protocol=4
                )

            if comm.rank == writer_rank:
                # Save electron density and intensity
                rho = np.fft.ifftshift(rho_)
                intensity = np.fft.ifftshift(np.abs(np.fft.fftshift(ac_phased) ** 2))
                save_mrc(settings.out_dir / f"ac-{generation}.mrc", ac_phased)
                save_mrc(settings.out_dir / f"intensity-{generation}.mrc", intensity)
                save_mrc(settings.out_dir / f"rho-{generation}.mrc", rho)
                
                if settings.checkpoint:
                    # Save output
                    myRes = {
                        **myRes,
                        **{
                            "ac_phased": ac_phased,
                            "support_": support_,
                            "rho_": rho_,
                            "orientations": orientations,
                        },
                    }
                    checkpoint.save_checkpoint(
                        myRes, settings.out_dir, generation, tag="", protocol=4
                    )

                # Check convergence w.r.t reference electron density
                if reference_dict["reference"] is not None:
                    prev_cc = final_cc
                    ali_volume, ali_reference, final_cc = align_volumes(
                        rho,
                        reference_dict["reference"],
                        zoom=settings.fsc_zoom,
                        sigma=settings.fsc_sigma,
                        n_iterations=settings.fsc_niter,
                        n_search=settings.fsc_nsearch,
                    )
                    resolution, rshell, fsc_val = compute_fsc(
                        ali_reference, ali_volume, reference_dict["dist_recip_max"]
                    )
                    delta_cc = final_cc - prev_cc
                    logger.log("Align volumes")
                    log_cuda_mem_info(logger)

            # Check if density converges
            if settings.chk_convergence:
                comm_compute = contexts.comm_compute  # safe for psana2
                resolution = comm_compute.bcast(resolution, root=0)
                final_cc = comm_compute.bcast(final_cc, root=0)
                delta_cc = comm_compute.bcast(delta_cc, root=0)
                logger.log(
                        f"Check convergence resolution: {resolution:.2f} with cc: {final_cc:.3f} delta_cc:{delta_cc:.5f}."
                )
                if final_cc > min_cc and delta_cc < min_change_cc:
                    logger.log(
                        f"Stopping criteria met! Algorithm converged at resolution: {resolution:.2f} with cc: {final_cc:.3f}."
                    )
                    flag_converged = True
                    ds.terminate()

            logger.log(f"Check convergence done in {timer.lap():.2f}s.")
            # Keeps record of last seen slice and reset processed event counter
            # (will be updated when N_images_per_rank is met)
            last_seen_slice = cn_processed_events - 1
            cn_processed_events = 0
            
            # A hack to free gpu memory at the end of a generation. Note that we only
            # need to recreate all these objects when no. of max images hasn't been reached.
            logger.log("Done")
            log_cuda_mem_info(logger)
            if settings.use_cuda:
                if last_seen_slice + 1 < N_images_max:
                    logger.log("Free GPUArrays and cufinufft plans")
                    nufft.free_gpuarrays_and_cufinufft_plans()
                    log_cuda_mem_info(logger)
                    logger.log("Free cupy memory pools") 
                    del nufft
                    del mg
                    del snm
                    gc.collect()
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    log_cuda_mem_info(logger)
                    # Set nufft to None since we haven't reached maximum no. of images so
                    # new nufft, etc. can be reallocated in the next generation.
                    nufft = None


            logger.log(f"Free memory done in {timer.lap():.2f}s.")
            # Update generation
            generation += 1

        # end for i_evt and (i_evt...

    # end for i_evt, evt in ...

    if settings.chk_convergence and comm.rank==writer_rank:
        msg = f"chk_convergence flag was set and the algorithm did no converge ({settings.fsc_min_cc=}, {settings.fsc_min_change_cc=})."
        assert flag_converged, msg
    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")
