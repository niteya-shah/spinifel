import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skew
import matplotlib.pyplot as plt
import PyNVTX as nvtx
import pygion
from pygion import task, Region, RO, RW, WD, Tunable, Partition, Region, execution_fence

from spinifel import settings
from spinifel.prep import save_mrc
from spinifel.sequential.phasing import phase as sequential_phase
from . import utils as lgutils


@nvtx.annotate("legion/phasing.py", is_prefix=True)
def create_phased_regions(N_procs, phased=None):
    summary_phase = Region(
        (N_procs,),
        {
            "rank": pygion.int32,
            "skews": pygion.float64,
            "height": pygion.float64,
            "width": pygion.float64,
        },
    )

    summary_phase_p = Partition.equal(summary_phase, (N_procs,))
    M = settings.M
    if settings.use_single_prec:
        ftype = pygion.float32
    else:
        ftype = pygion.float64
    multi_phased = Region(
        (N_procs * M, M, M),
        {"ac": ftype, "support_": pygion.bool_, "rho_": pygion.float64},
    )
    multi_phased_p = Partition.restrict(
        multi_phased, (N_procs,), [[M], [0], [0]], [M, M, M]
    )
    if phased is None:
        phased = Region(
            (settings.M,) * 3,
            {"ac": ftype, "support_": pygion.bool_, "rho_": pygion.float64},
        )
    # make sure phased regions are valid
    execution_fence(block=True)
    return {
        "summary": summary_phase,
        "summary_part": summary_phase_p,
        "multi_phase": multi_phased,
        "multi_phase_part": multi_phased_p,
        "phased": phased,
    }


@task(leaf=True, privileges=[WD("prev_rho_"), RO("rho_")])
@lgutils.gpu_task_wrapper
def prev_phase_task(prev_phased, phased):
    prev_phased.prev_rho_[:] = phased.rho_[:]


# select the partition index with the best result
@task(leaf=True, privileges=[RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def new_phase_select(summary):
    iref = np.argmin(summary.skews)
    ref_rank = summary.rank[iref]
    if settings.verbosity > 0:
        print(
            f"Keeping result from rank {ref_rank}: skew={summary.skews[ref_rank]:.2f}",
            flush=True,
        )
    return ref_rank


# use the best result
@task(leaf=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_result_task(results_p0, results, iref):
    results_p0.ac[:] = results.ac[iref.get(), :]
    results_p0.support_[:] = results.support_[iref.get(), :]
    results_p0.rho_[:] = results.rho_[iref.get(), :]


@task(leaf=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_result_subregion(results_p0, results):
    results_p0.ac[:] = results.ac[:]
    results_p0.support_[:] = results.support_[:]
    results_p0.rho_[:] = results.rho_[:]


# partition the region and use only the iref.get() subregion
@task(inner=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_result_task2(results_p0, results, results_p, iref):
    indx = iref.get()
    phased_result_subregion(results_p0, results_p[indx])


# new phase algorithm
def new_phase_gauss(rho_):
    _temp = rho_[np.where(rho_ > 0)]
    ydata, bins, _ = plt.hist(_temp.ravel(), bins=200)
    xdata = bins[:-1]  # resize bins to match ydata
    # Define the Gaussian function
    def Gauss(x, A, B, C):
        eps = 1e-9
        y = A * np.exp(-1 * (x - B) ** 2 / (2 * (C + eps) ** 2))
        return y

    try:
        parameters, covariance = curve_fit(Gauss, xdata, ydata)
    except RuntimeError:
        parameters = [0, 0, 0]

    fit_A = parameters[0]  # height of the curve's peak
    fit_B = parameters[1]  # the position of the center of the peak
    fit_C = parameters[2]  # the standard deviation
    return ydata, fit_C


@task(leaf=True, privileges=[RO("ac"), WD, WD])
@lgutils.gpu_task_wrapper
def new_phase_gen0_task(solved, phased_part, summary, generation, idx, num_procs):
    if settings.verbosity > 0:
        print("Starting phasing", flush=True)
    # shrinkwrap weight range: 0.5 to 1.5
    weight = 0.5 + idx / num_procs
    method = "std"
    phased_part.ac[:], phased_part.support_[:], phased_part.rho_[:] = sequential_phase(
        generation, solved.ac, None, None, method, weight
    )
    # TODO mpi/legion to reuse the function
    ydata, fit_c = new_phase_gauss(phased_part.rho_)
    summary.rank[0] = idx
    summary.height[0] = np.max(ydata)
    summary.width[0] = float(fit_c)
    summary.skews[0] = skew(ydata)
    if settings.verbosity > 0:
        print("Finishing phasing", flush=True)


@task(leaf=True, privileges=[RO("ac"), RO("support_", "rho_"), WD, WD])
@lgutils.gpu_task_wrapper
def new_phase_gen_task(
    solved, phased, summary, phased_part, generation, idx, num_procs
):
    if settings.verbosity > 0:
        print("Starting phasing", flush=True)
    # shrinkwrap weight range: 0.5 to 1.5
    weight = 0.5 + idx / num_procs
    method = "std"
    phased_part.ac[:], phased_part.support_[:], phased_part.rho_[:] = sequential_phase(
        generation, solved.ac, phased.support_, phased.rho_, method, weight
    )
    ydata, fit_c = new_phase_gauss(phased_part.rho_)
    summary.rank[0] = idx
    summary.height[0] = np.max(ydata)
    summary.width[0] = float(fit_c)
    summary.skews[0] = skew(ydata)
    if settings.verbosity > 0:
        print("Finishing phasing", flush=True)


@task(leaf=True, privileges=[RO("ac"), WD("ac", "support_", "rho_")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phase_gen0_task(solved, phased):
    if settings.verbosity > 0:
        print("Starting phasing", flush=True)
    phased.ac[:], phased.support_[:], phased.rho_[:] = sequential_phase(
        0, solved.ac, None, None
    )
    if settings.verbosity > 0:
        print("Finishing phasing", flush=True)


@nvtx.annotate("legion/phasing.py", is_prefix=True)
def create_phase_regions():
    num_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    phased_regions_dict = create_phased_regions(num_procs)
    return phased_regions_dict


@nvtx.annotate("legion/phasing.py", is_prefix=True)
def new_phase(generation, solved, phased_regions_dict=None):
    num_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    if generation == 0:
        if phased_regions_dict is None:
            phased_regions_dict = create_phased_regions(num_procs)
        multi_phased = phased_regions_dict["multi_phase_part"]
        summary = phased_regions_dict["summary_part"]
        phased_region = phased_regions_dict["phased"]
        for idx in range(num_procs):
            i = num_procs - idx - 1
            new_phase_gen0_task(
                solved, multi_phased[i], summary[i], generation, i, num_procs, point=i
            )
    else:
        assert phased_regions_dict is not None
        multi_phased = phased_regions_dict["multi_phase_part"]
        summary = phased_regions_dict["summary_part"]
        phased_region = phased_regions_dict["phased"]

        for idx in range(num_procs):
            i = num_procs - idx - 1
            new_phase_gen_task(
                solved,
                phased_region,
                summary[i],
                multi_phased[i],
                generation,
                i,
                num_procs,
                point=i,
            )

    summary_phase = phased_regions_dict["summary"]
    iref = new_phase_select(summary_phase)

    phased_region = phased_regions_dict["phased"]
    phased_all_region = phased_regions_dict["multi_phase"]
    phased_all_part = phased_regions_dict["multi_phase_part"]

    phased_result_task2(phased_region, phased_all_region, phased_all_part, iref)
    return phased_region, phased_regions_dict


@nvtx.annotate("legion/phasing.py", is_prefix=True)
def fill_phase_regions(phased_regions_dict):
    lgutils.fill_region(phased_regions_dict["summary"], 0)
    lgutils.fill_region(phased_regions_dict["multi_phase"], 0)
    lgutils.fill_region(phased_regions_dict["phased"], 0)


@task(leaf=True, privileges=[RO("ac", "rho_")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_output_task(phased, generation):
    rho = np.fft.ifftshift(phased.rho_)
    intensity = np.fft.ifftshift(np.abs(np.fft.fftshift(phased.ac) ** 2))
    save_mrc(settings.out_dir / f"ac-{generation}.mrc", phased.ac)
    save_mrc(settings.out_dir / f"rho-{generation}.mrc", rho)
    save_mrc(settings.out_dir / f"intensity-{generation}.mrc", intensity)


# launch the output task
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phased_output(phased, generation):
    phased_output_task(phased, generation)
