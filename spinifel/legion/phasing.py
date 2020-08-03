import numpy as np
import pygion
from pygion import task, Region, RO, RW, WD

from spinifel import parms
from spinifel.sequential.phasing import phase as sequential_phase



@task(privileges=[RO("ac"), WD("ac", "support_", "rho_")])
def phase_gen0_task(solved, phased):
    phased.ac[:] , phased.support_[:], phased.rho_[:] = sequential_phase(
            0, solved.ac, None, None)


@task(privileges=[RO("ac"), WD("ac") + RW("support_", "rho_")])
def phase_task(solved, phased, generation):
    phased.ac[:] , phased.support_[:], phased.rho_[:] = sequential_phase(
            generation, solved.ac, phased.support_, phased.rho_)


def phase(generation, solved, phased=None):
    Mquat = parms.Mquat
    M = 4*Mquat + 1
    Mtot = M**3

    if generation == 0:
        assert phased is None
        phased = Region((M,)*3, {
            "ac": pygion.float32, "support_": pygion.float32,
            "rho_": pygion.float32})
        phase_gen0_task(solved, phased)
    else:
        assert phased is not None
        phase_task(solved, phased, generation)

    return phased
