import numpy  as np
from scipy.optimize import curve_fit
from scipy.stats import skew
import matplotlib.pyplot as plt
import PyNVTX as nvtx
import pygion
from pygion import task, Region, RO, RW, WD, Tunable, Partition, Region

from spinifel import settings
from spinifel.prep import save_mrc
from spinifel.sequential.phasing import phase as sequential_phase
from . import utils as lgutils

@nvtx.annotate("legion/phasing.py", is_prefix=True)
def create_phased_regions(N_procs, phased=None):
    summary_phase = Region((N_procs,),
                     {"rank": pygion.int32, "skews": pygion.float32,
                      "height": pygion.int32,
                      "width": pygion.int32})

    summary_phase_p = Partition.equal(summary_phase, (N_procs,))
    M = settings.M
    multi_phased = Region((N_procs * M, M, M), {"ac": pygion.float32,
                                                "support_":pygion.float32,
                                                "rho_": pygion.float32})
    multi_phased_p = Partition.restrict(multi_phased, (N_procs,), [[M], [0], [0]], [M, M, M])
    if phased is None:
        phased = Region((settings.M,)*3, {
            "ac": pygion.float32,
            "support_": pygion.float32,
            "rho_": pygion.float32 }
        )
    return {'summary': summary_phase,
            'summary_part': summary_phase_p,
            'multi_phase': multi_phased,
            'multi_phase_part': multi_phased_p,
            'phased': phased }


@task(leaf=True, privileges=[WD("prev_rho_"), RO("rho_")])
@lgutils.gpu_task_wrapper
def prev_phase_task(prev_phased, phased):
    prev_phased.prev_rho_[:] = phased.rho_[:]

def prev_phase(generation, phased, prev_phased=None):
    assert phased is not None
    if generation == 1:
        assert prev_phased is None
        prev_phased = Region((settings.M,)*3, {
            "prev_rho_": pygion.float32,})
        prev_phase_task(prev_phased, phased)
    else:
        assert prev_phased is not None
        prev_phase_task(prev_phased, phased)
    return prev_phased


@task(leaf=True, privileges=[RO("prev_rho_"), RO("rho_")])
@lgutils.gpu_task_wrapper
def cov_task(prev_phased, phased, cov_xy, cov_delta):
    cc_matrix = np.corrcoef(prev_phased.prev_rho_.flatten(),
                            phased.rho_.flatten())
    val = cc_matrix[0,1]
    return val

def cov(prev_phased, phased, cov_xy, cov_delta):
    assert prev_phased is not None
    assert phased is not None
    fval = cov_task(prev_phased, phased, cov_xy, cov_delta)
    val = fval.get()
    is_cov = val - cov_xy < cov_delta
    return val, is_cov


#select the partition index with the best result
@task(leaf=True, privileges=[RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def new_phase_select(summary):
    iref = np.argmin(summary.skews)
    ref_rank = summary.rank[iref]
    if settings.verbosity > 0:
        print(f"height: {summary.height[ref_rank]}", flush=True)
        print(f"width: {summary.width[ref_rank]}", flush=True)
        print(f"skews: {summary.skews[ref_rank]}", flush=True)
        print(f"Keeping result from rank {ref_rank}: skew={summary.skews[ref_rank]:.2f}", flush=True)
    return ref_rank

#use the best result
@task(leaf=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_result_task(results_p0, results, iref):
    results_p0.ac[:] = results.ac[iref.get(),:]
    results_p0.support_[:] = results.support_[iref.get(),:]
    results_p0.rho_[:] = results.rho_[iref.get(),:]


@task(leaf=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_result_subregion(results_p0, results):
    results_p0.ac[:] = results.ac[:]
    results_p0.support_[:] = results.support_[:]
    results_p0.rho_[:] = results.rho_[:]

#partition the region and use only the iref.get() subregion
@task(inner=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_result_task2(results_p0, results, results_p, iref):
    indx = iref.get()
    phased_result_subregion(results_p0, results_p[indx])

# new phase algorithm
def new_phase_gauss(rho_):
    _temp = rho_[np.where(rho_>0)]
    ydata, bins, _ = plt.hist(_temp.ravel(),bins=200)
    xdata = bins[:-1] # resize bins to match ydata
    # Define the Gaussian function
    def Gauss(x, A, B, C):
        eps = 1e-9
        y = A*np.exp(-1*(x-B)**2/(2*(C+eps)**2))
        return y
    try:
        parameters, covariance = curve_fit(Gauss, xdata, ydata)
    except RuntimeError:
        parameters = [0, 0, 0]

    fit_A = parameters[0] # height of the curve's peak
    fit_B = parameters[1] # the position of the center of the peak
    fit_C = parameters[2] # the standard deviation
    return ydata, fit_C


@task(leaf=True, privileges=[RO("ac"),WD,WD])
@lgutils.gpu_task_wrapper
def new_phase_gen0_task(solved, phased_part, summary, generation, idx, num_procs):
    if settings.verbosity > 0:
        print("Starting phasing", flush=True)
    # shrinkwrap weight range: 0.5 to 1.5
    weight = 0.5 + idx/num_procs
    method = 'std'
    #TODO after the merge -> pick up the new sequential_phase method
    phased_part.ac[:], phased_part.support_[:], phased_part.rho_[:] = sequential_phase(generation, solved.ac, None, None)
    #TODO mpi/legion to reuse the function
    ydata, fit_c = new_phase_gauss(phased_part.rho_)
    summary.rank[0] = idx
    summary.height[0] = np.max(ydata)
    summary.width[0] = fit_c
    summary.skews[0] = skew(ydata)
    if settings.verbosity > 0:
        print("Finishing phasing", flush=True)

@task(leaf=True, privileges=[RO("ac"), RO("support_","rho_"), WD, WD])
@lgutils.gpu_task_wrapper
def new_phase_gen_task(solved, phased, summary, phased_part, generation, idx, num_procs):
    if settings.verbosity > 0:
        print("Starting phasing", flush=True)
    #shrinkwrap weight range: 0.5 to 1.5
    weight = 0.5 + idx/num_procs
    method = 'std'
    #TODO after the merge since to pick up the new sequential_phase method
    #ac_phased, support_, rho_ = sequential_phase(generation, ac, support_, rho_, method, weight)
    phased_part.ac[:], phased_part.support_[:], phased_part.rho_[:] = sequential_phase(generation, solved.ac, phased.support_, phased.rho_)
    ydata, fit_c = new_phase_gauss(phased_part.rho_)
    summary.rank[0] = idx
    summary.height[0] = np.max(ydata)
    summary.width[0] = fit_c
    summary.skews[0] = skew(ydata)
    if settings.verbosity > 0:
        print("Finishing phasing", flush=True)

@task(leaf=True, privileges=[RO("ac"), WD("ac", "support_", "rho_")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phase_gen0_task(solved, phased):
    if settings.verbosity > 0:
        print("Starting phasing", flush=True)
    phased.ac[:] , phased.support_[:], phased.rho_[:] = sequential_phase(
            0, solved.ac, None, None)
    if settings.verbosity > 0:
        print("Finishing phasing", flush=True)


@task(leaf=True, privileges=[RO("ac"), WD("ac") + RW("support_", "rho_")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/ phasing.py", is_prefix=True)
def phase_task(solved, phased, generation):
    if settings.verbosity > 0:
        print("Starting phasing", flush=True)
    phased.ac[:] , phased.support_[:], phased.rho_[:] = sequential_phase(
            generation, solved.ac, phased.support_, phased.rho_)
    if settings.verbosity > 0:
        print("Finishing phasing", flush=True)


@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phase(generation, solved, phased=None):
    if generation == 0:
        assert phased is None
        phased = Region((settings.M,)*3, {
            "ac": pygion.float32, "support_": pygion.float32,
            "rho_": pygion.float32})
        phase_gen0_task(solved, phased)
    else:
        assert phased is not None
        phase_task(solved, phased, generation)

    return phased

@nvtx.annotate("legion/phasing.py", is_prefix=True)
def new_phase(generation, solved, phased_regions_dict=None):
    num_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    if generation == 0:
        assert phased_regions_dict is None
        phased_regions_dict = create_phased_regions(num_procs)
        multi_phased = phased_regions_dict['multi_phase_part']
        summary = phased_regions_dict['summary_part']
        phased_region = phased_regions_dict['phased']
        for i in range(num_procs):
            new_phase_gen0_task(solved, multi_phased[i], summary[i], generation, i, num_procs,point=i)
    else:
        assert phased_regions_dict is not None
        multi_phased = phased_regions_dict['multi_phase_part']
        summary = phased_regions_dict['summary_part']
        phased_region = phased_regions_dict['phased']

        for i in range(num_procs):
            new_phase_gen_task(solved, phased_region, summary[i],  multi_phased[i], generation, i, num_procs, point=i)

    summary_phase = phased_regions_dict['summary']
    iref = new_phase_select(summary_phase)

    phased_region = phased_regions_dict['phased']
    phased_all_region = phased_regions_dict['multi_phase']
    phased_all_part = phased_regions_dict['multi_phase_part']

    phased_result_task2(phased_region, phased_all_region, phased_all_part, iref)
    return phased_region, phased_regions_dict

@nvtx.annotate("legion/phasing.py", is_prefix=True)
def fill_phase_regions(phased_regions_dict):
    lgutils.fill_region(phased_regions_dict['summary'])
    lgutils.fill_region(phased_regions_dict['multi_phase'])
    lgutils.fill_region(phased_regions_dict['phased'])

@task(leaf=True, privileges=[RO("ac", "rho_")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_output_task(phased,generation):
    rho = np.fft.ifftshift(phased.rho_)
    save_mrc(settings.out_dir / f"ac-{generation}.mrc", phased.ac)
    save_mrc(settings.out_dir / f"rho-{generation}.mrc", rho)

# launch the output task
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_output(phased,generation):
    phased_output_task(phased, generation)
