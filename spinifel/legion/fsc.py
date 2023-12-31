import numpy  as np
import PyNVTX as nvtx
import pygion
import math
from pygion import task, RO
from spinifel import settings
from spinifel import utils
from . import utils as lgutils
from eval.fsc import compute_fsc, compute_reference
from eval.align import align_volumes
if settings.use_cupy:
    import cupy

@task(privileges=[RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/fsc.py", is_prefix=True)
def init_fsc_task(pixel_distance):
    fsc = {}
    if settings.verbose:
        print(f"started init_fsc Task", flush=True)
        
    dist_recip_max = np.max(pixel_distance.reciprocal)
    fsc['reference'] = compute_reference(
        settings.pdb_path, settings.M,
        dist_recip_max)
    fsc['final'] = 0.0
    fsc['delta'] = 1.0
    fsc['res'] = 0.0
    fsc['min_cc'] = settings.fsc_min_cc
    fsc['min_change_cc'] = settings.fsc_min_change_cc
    fsc['dist_recip_max'] = dist_recip_max
    fsc['converge'] = False

    if settings.verbose:
        print(f"finished init_fsc Task", flush=True)
    return fsc

@task(privileges=[RO('rho_')])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/fsc.py", is_prefix=True)
def compute_fsc_task(phased, fsc):
    if settings.verbose:
        timer = utils.Timer()
    fsc_dict = fsc.get()
    prev_cc = fsc_dict['final']
    rho = np.fft.ifftshift(phased.rho_)
    ali_volume, ali_reference, final_cc = align_volumes(rho,
                                                        fsc_dict['reference'],
                                                        zoom=settings.fsc_zoom,
                                                        sigma=settings.fsc_sigma,
                                                        n_iterations=settings.fsc_niter,
                                                        n_search=settings.fsc_nsearch)
    resolution, rshell, fsc_val = compute_fsc(ali_reference, ali_volume, fsc_dict['dist_recip_max'])
    # uses a lot of memory - release it asap
    if settings.use_cupy and settings.cupy_mempool_clear:
        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()

    if settings.verbose:
        print(f'FSC clear_cupy_mempool:{settings.cupy_mempool_clear} completed in: {timer.lap():.2f}s.', flush=True)

    min_cc = fsc_dict['min_cc']
    delta_cc = final_cc - prev_cc
    min_change_cc = fsc_dict['min_change_cc']
    fsc_dict['delta'] = delta_cc
    fsc_dict['res'] = resolution
    fsc_dict['final'] = final_cc
    # no change in vals
    if math.isclose(final_cc, min_cc) and math.isclose(delta_cc, min_change_cc):
        return fsc_dict
    else:
        if final_cc > min_cc and delta_cc < min_change_cc:
            fsc_dict['converge'] = True
            print(f"Stopping criteria met! Algorithm converged at resolution: {resolution:.2f} with cc: {final_cc:.3f}.", flush=True)
    return fsc_dict

@task
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/fsc.py", is_prefix=True)
def check_convergence_task(fsc):
    converge = False
    fsc_dict = fsc.get()
    if fsc_dict['converge'] == True:
        converge = True
    return converge
