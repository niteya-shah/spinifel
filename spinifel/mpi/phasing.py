import numpy as np

from spinifel import parms, contexts
from spinifel.sequential.phasing import phase as sequential_phase


def phase(generation, ac, support_=None, rho_=None):
    """Phase retrieval by Rank0 and broadcast to all ranks."""

    Mquat = parms.Mquat
    M = 4*Mquat + 1
    Mtot = M**3

    ac_phased, support_, rho_ = sequential_phase(
        generation, ac, support_, rho_)

    return ac_phased, support_, rho_
