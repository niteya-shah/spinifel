from spinifel import settings, utils, contexts, checkpoint, image
from spinifel.prep import (
    save_mrc,
    compute_pixel_distance,
    binning_mean,
    binning_index,
    load_pixel_position_reciprocal_psana,
)

import numpy as np
import PyNVTX as nvtx
import os

from .prep import compute_mean_image, show_image, bin_data, get_pixel_info, get_data
from .phasing import phase

from spinifel.mpi.orientation_matching import SNM_MPI
from .autocorrelation import MergeMPI
from spinifel.extern.nufft_ext_mpi import NUFFT_MPI

from eval.fsc import compute_fsc, compute_reference
from eval.align import align_volumes

# For main and unit tests
from .test_util import get_known_orientations

# For making sure that gpu memory is released.
import gc

import datetime

if settings.use_cuda:
    if not settings.use_pygpu:
        import pycuda.driver as cuda
    import cupy

    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()


def log_cuda_mem_info(logger):
    if settings.use_cuda:
        if not settings.use_pygpu:
            (free, total) = cuda.mem_get_info()
            logger.log(
                f"Global memory occupancy: {free*100/total:.2f}% free ({free/1e9:.2f}/{total/1e9:.2f} GB)", 
                level=1
            )
        mempool_used = mempool.used_bytes() * 1e-9
        mempool_total = mempool.total_bytes() * 1e-9
        logger.log(
            f"|-->Cupy: {mempool_used=:.2f}GB {mempool_total=:.2f}GB {pinned_mempool.n_free_blocks()=:d}",
            level=1
        )


class Container:
    def __init__(self):
        pass

class EventManager:
    def __init__(self, run=None,photon=False):
        self.run = run
        self.photon = photon
        self.det = None
        if run is not None:
            self.det = run.Detector(settings.ps_detname)
        self.h5py_read_number = 0
    def events(self):
        if self.run is not None:
            for evt in self.run.events():
                evt_cner = Container()
                img = self.det.raw.calib(evt)
                setattr(evt_cner, 'slice', img) 
                setattr(evt_cner, 'timestamp', evt.timestamp)
                if self.photon:
                    photon_energy = self.det.raw.photon_energy(evt)
                    setattr(evt_cner, 'photon_energy', photon_energy)
                yield evt_cner
        else:
            N_images_per_rank = settings.N_images_per_rank
            while True:
                if N_images_per_rank * self.h5py_read_number < settings.N_images_max:
                    slices_ = get_data(N_images_per_rank, read_number=self.h5py_read_number)
                # Increment read number so we read the next N_images_per_rank for this rank
                self.h5py_read_number +=1       
                for slice_ in slices_:
                    evt_cner = Container()
                    setattr(evt_cner, 'slice', slice_) 
                    yield evt_cner


@nvtx.annotate("mpi/main.py", is_prefix=True)
def main():
    comm = contexts.comm

    timer = utils.Timer()

    N_images_per_rank = settings.N_images_per_rank
    N_images_max = settings.N_images_max
    assert (
        N_images_max % N_images_per_rank == 0
    ), "N_images_max must be divisible by N_images_per_rank"

    N_generations = settings.N_generations

    # BigData cores are those excluding Smd0, EventBuilder, & Server cores.
    N_big_data_nodes = comm.size - (1 + settings.ps_eb_nodes + settings.ps_srv_nodes)

    # Reading input images using psana2 or h5py
    if settings.use_psana:
        from psana import DataSource
        ds = DataSource(
            exp=settings.ps_exp,
            run=settings.ps_runnum,
            dir=settings.ps_dir,
            batch_size=settings.ps_batch_size,
        )
        run = next(ds.runs())
        # Writer rank is the first bigdata core
        writer_rank = 1 + settings.ps_eb_nodes
    else:
        ds = None
        run = None
        writer_rank = 0

    # Setup logger for all worker ranks
    logger = utils.Logger(contexts.is_worker, settings, myrank=comm.rank)
    logger.log("In MPI main")
    if settings.use_psana:
        logger.log("Using psana")
        logger.log(f"PS_SMD_N_EVENTS: {os.environ.get('PS_SMD_N_EVENTS','10000')}")
        logger.log(f"ps batch_size  : {settings.ps_batch_size}")
        logger.log(f"#bdcores       : {N_big_data_nodes:d}")
    logger.log(f"comm.size      : {comm.size:d}")
    logger.log(f"writerrank     : {writer_rank}")
    logger.log(f"#img/rank      : {N_images_per_rank}")

    # Skip this data saving and ac calculation in test mode
    generation = 0
    reference_dict = {}
    if settings.load_gen > 0:  # Load input from previous generation
        generation = settings.load_gen
        logger.log(
            f"Loading checkpoint: {checkpoint.generate_checkpoint_name(settings.out_dir, settings.load_gen, settings.tag_gen)}",
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

    # Create event generator from either psana2 ds or h5py
    evt_man = EventManager(run)
    
    # Obtain run-related info.
    (pixel_position_reciprocal, pixel_index_map, pixel_position) = get_pixel_info(run)
    raw_pixel_position_reciprocal = np.zeros(pixel_position_reciprocal.shape, dtype=pixel_position_reciprocal.dtype)
    raw_pixel_position_reciprocal[:] = pixel_position_reciprocal
    raw_pixel_index_map = np.zeros(pixel_index_map.shape, dtype=pixel_index_map.dtype)
    raw_pixel_index_map[:] = pixel_index_map

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
    for i_evt, evt in enumerate(evt_man.events()):
        # Quit reading when max generations reached
        if generation == N_generations:
            if settings.use_psana:
                ds.terminate()
            else:
                break

        # Only need to do once for data that needs to convert pixel_position
        if pixel_position_reciprocal is None:
            photon_energy =  evt.photon_energy

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
            raw_pixel_position_reciprocal = np.zeros(pixel_position_reciprocal.shape, dtype=pixel_position_reciprocal.dtype)
            raw_pixel_position_reciprocal[:] = pixel_position_reciprocal

        # Start collecting slices only until max and count no. of processed
        # images. This no. is no longer increased when i_evt exceeds N_images_max.
        if i_evt < N_images_max:
            raw_slices_[cn_new_events] = evt.slice #det.raw.calib(evt)
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
            logger.log(f"datetime.datetime.now() = {str(datetime.datetime.now())}")
            logger.log(f"#" * 45)
            logger.log(
                f"##### Generation {generation}/{N_generations} Slices:{cn_processed_events}/{N_images_max} #####"
            )
            logger.log(f"#" * 45)
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
                nufft = NUFFT_MPI(
                    settings,
                    pixel_position_reciprocal,
                    pixel_distance_reciprocal,
                    cn_processed_events,
                )
                mg = MergeMPI(
                    settings,
                    slices_,
                    pixel_position_reciprocal,
                    pixel_distance_reciprocal,
                    nufft,
                )

                snm = SNM_MPI(
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
                ac = mg.solve_ac(generation)
                logger.log(f"AC recovered in {timer.lap():.2f}s.")

                # If the pdb file is given, the writer rank will calculate this
                reference, dist_recip_max = (None, None)
                if (
                    settings.pdb_path.is_file()
                    and settings.chk_convergence
                    and comm.rank == writer_rank
                ):
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
                        "reference": reference,
                        "pixel_position_reciprocal": pixel_position_reciprocal,
                        "pixel_distance_reciprocal": pixel_distance_reciprocal,
                        "slices_": slices_,
                        "ac": ac,
                    }
                    checkpoint.save_checkpoint(
                        myRes, settings.out_dir, generation, tag="solve_ac_init", protocol=4
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
                        myRes, settings.out_dir, generation, tag="phase_init", protocol=4
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

            ############################################################################
            # Slice and Orientation matching
            orientations = snm.slicing_and_match(ac_phased)

            # In test mode, we supply some correct orientations to guarantee convergence
            if settings.fsc_fraction_known_orientations > 0 and generation == 0:
                N_supply = int(
                    settings.fsc_fraction_known_orientations * orientations.shape[0]
                )
                comm_compute = contexts.comm_compute  # safe for psana2
                known_orientations = get_known_orientations()
                i_start = comm_compute.rank * N_images_per_rank
                i_end = i_start + N_supply
                logger.log(
                    f"****WARNING**** In Test Mode - supplying {i_start}-{i_end} correct orientations"
                )
                orientations[:N_supply,:] = known_orientations[i_start:i_end,:]

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
                    f"Check convergence resolution: {resolution:.2f} with cc: {final_cc:.3f} delta_cc:{delta_cc:.5f}.", level=1
                )
                if final_cc > min_cc and delta_cc < min_change_cc:
                    logger.log(
                        f"Stopping criteria met! Algorithm converged at resolution: {resolution:.2f} with cc: {final_cc:.3f}.", level=1
                    )
                    flag_converged = True
                    if settings.use_psana:
                        ds.terminate()
                    else:
                        break

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
                    logger.log("Free GPUArrays and cufinufft plans", level=1)
                    nufft.free_gpuarrays_and_cufinufft_plans()
                    log_cuda_mem_info(logger)
                    logger.log("Free cupy memory pools", level=1)
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

            logger.log(f"Free memory done in {timer.lap():.2f}s.", level=1)
            # Update generation
            generation += 1

        # end for i_evt and (i_evt...

    # end for i_evt, evt in ...

    if settings.chk_convergence and comm.rank == writer_rank:
        msg = f"chk_convergence flag was set and the algorithm did no converge ({settings.fsc_min_cc=}, {settings.fsc_min_change_cc=})."
        assert flag_converged, msg
    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")
