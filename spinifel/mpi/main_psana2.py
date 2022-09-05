from spinifel import settings, utils, contexts, checkpoint, image
from spinifel.prep import save_mrc, compute_pixel_distance, binning_mean, binning_index

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


def get_known_answers(
    logger, mg, pixel_position_reciprocal, pixel_distance_reciprocal, slices
):
    """Returns known answers For unit-test [DO NOT REMOVE]

    This main is also called directly by Spinifel's main test.
    The test is done only for 3iyf (see settings/test_mpi.toml)
    and we get the known orientations and ac_phased directly
    from test_data_dir folder for comparisons.
    """
    test_data_dir = os.environ.get("test_data_dir", "")
    N_test_orientations = settings.N_orientations

    # Open data file with correct answers
    import h5py

    test_data = h5py.File(os.path.join(test_data_dir, "3IYF", "3iyf_sim_10k.h5"), "r")

    # Get known orientations
    ref_orientations = test_data["orientations"][:N_test_orientations]
    ref_orientations = np.reshape(ref_orientations, [N_test_orientations, 4])

    # Calculate ac_phased
    # Here volume (in test_data) is the fourier amplitudes. The ivol is
    # the intensity.

    ivol = np.square(np.abs(test_data["volume"]))
    known_ac_phased = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))).astype(np.float32)

    # Calculate rho from correct orientations (as known rho)
    generation = 0
    known_ac = mg.solve_ac(
        generation,
        orientations=ref_orientations[: slices.shape[0]],
    )

    # ISSUE_51: Uncomment below for the old solve ac prior to the mg use above
    # known_ac = work_solve_ac(
    #    generation, pixel_position_reciprocal, pixel_distance_reciprocal,
    #    slices, ref_orientations[:slices.shape[0]])

    _, _, known_rho = phase(generation, known_ac)

    logger.log(
        f"[Warning] - test mode ref_orientations:{ref_orientations.shape} known_ac_phased:{known_ac_phased.shape} known_rho:{known_rho.shape}"
    )
    return ref_orientations, known_ac_phased, known_rho


@nvtx.annotate("mpi/main.py", is_prefix=True)
def main():
    comm = contexts.comm

    timer = utils.Timer()

    # Reading input images from hdf5
    N_images_per_rank = settings.N_images_per_rank
    N_images_max = settings.N_images_max
    # batch_size = min(N_images_per_rank, 100)
    N_big_data_nodes = comm.size
    # max_events = min(settings.N_images_max, N_big_data_nodes*N_images_per_rank)
    writer_rank = 0  # pick writer rank as core 0
    N_generations = settings.N_generations

    assert settings.use_psana

    # Reading input images using psana2
    from psana import DataSource

    # BigData cores are those excluding Smd0, EventBuilder, & Server cores.
    N_big_data_nodes = comm.size - (1 + settings.ps_eb_nodes + settings.ps_srv_nodes)
    writer_rank = 1 + settings.ps_eb_nodes  # pick writer rank as the first BigData core
    ds = DataSource(exp=settings.ps_exp, run=settings.ps_runnum, dir=settings.ps_dir)

    # Setup logger after knowing the writer rank
    logger = utils.Logger(comm.rank == writer_rank)
    logger.log("In MPI main")
    if settings.use_psana:
        logger.log("Using psana")
    logger.log(f"comm.size : {comm.size:d}")
    logger.log(f"#workers  : {N_big_data_nodes:d}")
    logger.log(f"writerrank: {writer_rank}")
    logger.log(f"#img/rank : {N_images_per_rank}")

    # Skip this data saving and ac calculation in test mode
    generation = 0
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

    # Use improvement of cc(prev_rho, cur_rho) to dertemine if we should
    # terminate the loop
    min_cc, min_change_cc = 0.80, 0.001
    final_cc, delta_cc = 0.0, 1.0
    resolution = 0.0

    # Obtain run-related info.
    run = next(ds.runs())
    det = run.Detector("amopnccd")

    _pixel_index_map = run.beginruns[0].scan[0].raw.pixel_index_map
    pixel_index_map = np.moveaxis(_pixel_index_map[:], -1, 0)
    raw_pixel_index_map = pixel_index_map

    pixel_position_reciprocal = None
    if hasattr(run.beginruns[0].scan[0].raw, "pixel_position_reciprocal"):
        _pixel_position_reciprocal = (
            run.beginruns[0].scan[0].raw.pixel_position_reciprocal
        )
        pixel_position_reciprocal = np.moveaxis(_pixel_position_reciprocal[:], -1, 0)
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
    flag_test = False

    # Looping over events and run spinifel when receive enough events
    for i_evt, evt in enumerate(run.events()):
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

            logger.log(f"Loaded in {timer.lap():.2f}s.")
            logger.log(f"#" * 42)
            logger.log(
                f"##### Generation {generation}/{N_generations} Slices:{cn_processed_events}/{N_images_max}#####"
            )
            logger.log(f"#" * 42)

            # Intitilize merge class - must be done before get_known_answers (mg needed)
            nufft = NUFFT(
                settings, pixel_position_reciprocal, pixel_distance_reciprocal
            )
            mg = MergeMPI(
                settings,
                slices_,
                pixel_position_reciprocal,
                pixel_distance_reciprocal,
                nufft,
            )

            # For unit test [DO NOT REMOVE]
            if generation == 0:
                ref_orientations = None
                if os.environ.get("SPINIFEL_TEST_MODULE", "") == "MAIN_PSANA2":
                    flag_test = True
                    test_accept_thres = 0.75
                    ref_orientations, known_ac_phased, known_rho = get_known_answers(
                        logger,
                        mg,
                        pixel_position_reciprocal,
                        pixel_distance_reciprocal,
                        slices_,
                    )

            # Initialize orientation matching class must be done after get_known_answers
            # to obtain ref_orientations
            snm = SNM(
                settings,
                slices_,
                pixel_position_reciprocal,
                pixel_distance_reciprocal,
                nufft,
            )

            # Solve autocorrelation first for generation 0
            if generation == 0:
                # ac = work_solve_ac(
                #    generation, pixel_position_reciprocal, pixel_distance_reciprocal, slices_)
                ac = mg.solve_ac(generation)
                logger.log(f"AC recovered in {timer.lap():.2f}s.")

                # If the pdb file is given and checkpoint is set, the writer rank will calculate this
                if settings.checkpoint and comm.rank == writer_rank:
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
                        myRes, settings.out_dir, generation, tag="solve_ac", protocol=4
                    )

                ac_phased, support_, rho_ = phase(generation, ac)
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
                        myRes, settings.out_dir, generation, tag="phase", protocol=4
                    )
                    # Save electron density and intensity
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
            #    slices_, orientations, ac_phased)
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

            # Test correlation of the known vs calculated electron density
            if flag_test:
                cc_test_rho = np.corrcoef(known_rho.flatten(), rho_.flatten())[0, 1]
                logger.log(
                    f"[Warning] test mode cc(known_rho, rho_):{cc_test_rho} !! assert disabled !!"
                )
                # assert cc_test_rho > test_accept_thres

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

            # Check if density converges
            if settings.chk_convergence:
                comm_compute = contexts.comm_compute  # safe for psana2
                resolution = comm_compute.bcast(resolution, root=0)
                final_cc = comm_compute.bcast(final_cc, root=0)
                delta_cc = comm_compute.bcast(delta_cc, root=0)
                logger.log(
                    f"Check convergence resolution: {resolution:.2f} with cc: {final_cc:.3f}."
                )
                if final_cc > min_cc and delta_cc < min_change_cc:
                    logger.log(
                        f"Stopping criteria met! Algorithm converged at resolution: {resolution:.2f} with cc: {final_cc:.3f}."
                    )
                    ds.terminate()

            # Keeps record of last seen slice and reset processed event counter
            # (will be updated when N_images_per_rank is met)
            last_seen_slice = cn_processed_events - 1
            cn_processed_events = 0

            # Update generation
            generation += 1

        # end for i_evt and (i_evt...

    # end for i_evt, evt in ...

    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")
