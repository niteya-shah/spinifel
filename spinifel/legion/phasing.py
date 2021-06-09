import numpy  as np
import PyNVTX as nvtx
import pygion
from pygion import task, Region, RO, RW, WD

from spinifel import parms
from spinifel.sequential.phasing import phase as sequential_phase

@task(privileges=[WD("prev_rho_"), RO("rho_")])
def prev_phase_task(prev_phased, phased):
    prev_phased.prev_rho_[:] = phased.rho_[:]

def prev_phase(generation, phased, prev_phased=None):
    assert phased is not None
    if generation == 1:
        assert prev_phased is None
        prev_phased = Region((parms.M,)*3, {
            "prev_rho_": pygion.float32,})
        prev_phase_task(prev_phased, phased)
    else:
        assert prev_phased is not None
        prev_phase_task(prev_phased, phased)
    return prev_phased


@task(privileges=[RO("prev_rho_"), RO("rho_")])
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

@task(privileges=[RO("ac"), WD("ac", "support_", "rho_")])
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phase_gen0_task(solved, phased):
    if parms.verbosity > 0:
        print("Starting phasing", flush=True)
    phased.ac[:] , phased.support_[:], phased.rho_[:] = sequential_phase(
            0, solved.ac, None, None)
    if parms.verbosity > 0:
        print("Finishing phasing", flush=True)



@task(privileges=[RO("ac"), WD("ac") + RW("support_", "rho_")])
@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phase_task(solved, phased, generation):
    if parms.verbosity > 0:
        print("Starting phasing", flush=True)
    phased.ac[:] , phased.support_[:], phased.rho_[:] = sequential_phase(
            generation, solved.ac, phased.support_, phased.rho_)
    if parms.verbosity > 0:
        print("Finishing phasing", flush=True)



@nvtx.annotate("legion/phasing.py", is_prefix=True)
def phase(generation, solved, phased=None):
    if generation == 0:
        assert phased is None
        phased = Region((parms.M,)*3, {
            "ac": pygion.float32, "support_": pygion.float32,
            "rho_": pygion.float32})
        phase_gen0_task(solved, phased)
    else:
        assert phased is not None
        phase_task(solved, phased, generation)

    return phased
