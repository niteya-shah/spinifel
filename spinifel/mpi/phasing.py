import numpy as np

from mpi4py import MPI

from spinifel import settings, contexts
from spinifel.sequential.phasing import phase as sequential_phase


def phase(generation, ac, support_=None, rho_=None):
    """Phase retrieval by Rank0 and broadcast to all ranks."""

    comm = MPI.COMM_WORLD
    Mquat = settings.Mquat
    M = 4 * Mquat + 1
    Mtot = M**3

    ac_phased, support_, rho_ = sequential_phase(
        generation, ac, support_, rho_)

    comm.Bcast(ac_phased, root=0)
    comm.Bcast(support_, root=0)
    comm.Bcast(rho_, root=0)

    return ac_phased, support_, rho_
