from spinifel import settings, utils, contexts, checkpoint, image
from spinifel.prep import save_mrc, compute_pixel_distance, binning_mean, binning_index

import numpy as np
import PyNVTX as nvtx
import os

from .prep import get_data, compute_mean_image, show_image, bin_data
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

@nvtx.annotate("mpi/main.py", is_prefix=True)
def main():
    comm = contexts.comm

    timer = utils.Timer()

    # Reading input images from hdf5
    N_images_per_rank = settings.N_images_per_rank
    batch_size = min(N_images_per_rank, 100)
    N_big_data_nodes = comm.size
    max_events = min(settings.N_images_max, N_big_data_nodes * N_images_per_rank)
    writer_rank = 0  # pick writer rank as core 0

    # Reading input images using psana2
    ds = None
    if settings.use_psana:
        from psana import DataSource

        # BigData cores are those excluding Smd0, EventBuilder, & Server cores.
        N_big_data_nodes = comm.size - (
            1 + settings.ps_eb_nodes + settings.ps_srv_nodes
        )
        writer_rank = (
            1 + settings.ps_eb_nodes
        )  # pick writer rank as the first BigData core

        # Limit batch size to 100
        batch_size = min(N_images_per_rank, 100)

        # Calculate total no. of images that will be processed (limit by max)
        # The +2 per rank is to cover special transitions for xtc2
        max_events = min(
            (N_big_data_nodes * settings.N_images_max),
            (N_big_data_nodes * N_images_per_rank) + 2,
        )

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
        ds = DataSource(
            exp=settings.ps_exp,
            run=settings.ps_runnum,
            dir=settings.ps_dir,
            destination=destination,
            max_events=max_events,
        )

    # Setup logger after knowing the writer rank
    logger = utils.Logger(comm.rank == writer_rank)
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
    (pixel_position_reciprocal, pixel_index_map, slices_) = get_data(
        N_images_per_rank, ds
    )
    logger.log(f"slices_: {slices_.shape}")

    # Hacky way to allow only worker ranks for computation tasks
    if not contexts.is_worker:
        return

    # Computes reciprocal distance and mean image then save to .png
    pixel_distance_reciprocal = compute_pixel_distance(pixel_position_reciprocal)
    mean_image = compute_mean_image(slices_)
    # This is only done on first worker rank - other ranks will see None for mean_image
    if mean_image is not None:
        show_image(
            image,
            ds,
            contexts.rank,
            slices_,
            pixel_index_map,
            pixel_position_reciprocal,
            pixel_distance_reciprocal,
            mean_image,
            "image_0.png",
            "mean_image.png",
            "saxs.png",
        )

    # Bins data and save to .png files
    (pixel_position_reciprocal, pixel_index_map, slices_) = bin_data(
        pixel_position_reciprocal, pixel_index_map, slices_
    )
    pixel_distance_reciprocal = compute_pixel_distance(pixel_position_reciprocal)
    mean_image = compute_mean_image(slices_)
    if mean_image is not None:
        show_image(
            image,
            ds,
            contexts.rank,
            slices_,
            pixel_index_map,
            pixel_position_reciprocal,
            pixel_distance_reciprocal,
            mean_image,
            "image_binned_0.png",
            "mean_image_binned.png",
            "saxs_binned.png",
        )

    logger.log(f"Loaded in {timer.lap():.2f}s.")

    # Generation 0: solve_ac and phase
    N_generations = settings.N_generations

    # Intitilize merge and orientation matching 
    nufft = NUFFT(settings, pixel_position_reciprocal, pixel_distance_reciprocal)
    mg = MergeMPI(
        settings, slices_, pixel_position_reciprocal, pixel_distance_reciprocal, nufft
    )

    snm = SNM(
        settings,
        slices_,
        pixel_position_reciprocal,
        pixel_distance_reciprocal,
        nufft,
    )

    if settings.load_gen > 0:  # Load input from previous generation
        curr_gen = settings.load_gen
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
    else:
        curr_gen = 0
        logger.log(f"#" * 27)
        logger.log(f"##### Generation {curr_gen}/{N_generations} #####")
        logger.log(f"#" * 27)

        ac = mg.solve_ac(curr_gen)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")
        if comm.rank == writer_rank and settings.checkpoint:
            reference = None
            dist_recip_max = None
            if settings.pdb_path.is_file():
                dist_recip_max = np.max(pixel_distance_reciprocal)
                reference = compute_reference(
                    settings.pdb_path, settings.M, dist_recip_max
                )
                logger.log(f"Reference created in {timer.lap():.2f}s.")
            myRes = {
                "pixel_position_reciprocal": pixel_position_reciprocal,
                "pixel_distance_reciprocal": pixel_distance_reciprocal,
                "slices_": slices_,
                "ac": ac,
                "reference": reference,
                "dist_recip_max": dist_recip_max,
            }
            checkpoint.save_checkpoint(
                myRes, settings.out_dir, curr_gen, tag="solve_ac", protocol=4
            )

        ac_phased, support_, rho_ = phase(curr_gen, ac)
        logger.log(f"Problem phased in {timer.lap():.2f}s.")

        if comm.rank == writer_rank:
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
                myRes, settings.out_dir, curr_gen, tag="phase", protocol=4
            )
            # Save electron density and intensity
            rho = np.fft.ifftshift(rho_)
            intensity = np.fft.ifftshift(np.abs(np.fft.fftshift(ac_phased) ** 2))
            save_mrc(settings.out_dir / f"ac-{curr_gen}.mrc", ac_phased)
            save_mrc(settings.out_dir / f"intensity-{curr_gen}.mrc", intensity)
            save_mrc(settings.out_dir / f"rho-{curr_gen}.mrc", rho)

    # Use improvement of cc(prev_rho, cur_rho) to dertemine if we should
    # terminate the loop
    min_cc, min_change_cc = settings.fsc_min_cc, settings.fsc_min_change_cc
    final_cc, delta_cc = 0.0, 1.0
    resolution = 0.0
    curr_gen += 1
    flag_converged = False

    for generation in range(curr_gen, N_generations + 1):
        logger.log(f"#" * 27)
        logger.log(f"##### Generation {generation}/{N_generations} #####")
        logger.log(f"#" * 27)
        # Orientation matching
        # orientations = work_match(
        #    known_ac_phased, slices_,
        #    pixel_position_reciprocal,
        #    pixel_distance_reciprocal,)

        orientations = snm.slicing_and_match(ac_phased)
        
        # In test mode, we supply some correct orientations to guarantee convergence
        if int(os.environ.get("SPINIFEL_TEST_FLAG", "0")) and generation==1:
            logger.log(f"****WARNING**** In Test Mode - supplying some correct orientations")
            N_supply = int(.33 * orientations.shape[0])
            orientations[:N_supply] = get_known_orientations()[:N_supply]

        logger.log(f"Orientations matched in {timer.lap():.2f}s.")
        if comm.rank == writer_rank:
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
        #    slices_, orientations)
        ac = mg.solve_ac(generation, orientations, ac_phased)

        logger.log(f"AC recovered in {timer.lap():.2f}s.")
        if comm.rank == writer_rank:
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

            # Save rho and support for comparisons in the next generation
            prev_rho_ = rho_[:]
            prev_support_ = support_[:]

        ac_phased, support_, rho_ = phase(generation, ac, support_, rho_)

        logger.log(f"Problem phased in {timer.lap():.2f}s.")
        if comm.rank == writer_rank:
            myRes = {
                **myRes,
                **{
                    "ac": ac,
                    "prev_support_": prev_support_,
                    "prev_rho_": prev_rho_,
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
            if myRes["reference"] is not None:
                prev_cc = final_cc
                ali_volume, ali_reference, final_cc = align_volumes(
                    rho,
                    myRes["reference"],
                    zoom=settings.fsc_zoom,
                    sigma=settings.fsc_sigma,
                    n_iterations=settings.fsc_niter,
                    n_search=settings.fsc_nsearch,
                )
                resolution, rshell, fsc_val = compute_fsc(
                    ali_reference, ali_volume, myRes["dist_recip_max"]
                )
                delta_cc = final_cc - prev_cc

        if settings.chk_convergence:
            comm_compute = contexts.comm_compute
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
                break

    if settings.chk_convergence:
        msg = f"chk_convergence flag was set and the algorithm did no converge ({settings.fsc_min_cc=}, {settings.fsc_min_change_cc=})."
        assert flag_converged, msg
    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")
