import numpy as np
from mpi4py import MPI

from spinifel import parms, contexts
from spinifel.sequential.phasing import phase as sequential_phase


def phase(generation, ac, support_=None, rho_=None):
    """Phase retrieval by Rank0 and broadcast to all ranks."""
    comm = MPI.COMM_WORLD

    Mquat = parms.Mquat
    M = 4*Mquat + 1
    Mtot = M**3

    if comm.rank == 0:
        ac_phased, support_, rho_ = sequential_phase(
            generation, ac, support_, rho_)
    else:
        ac_phased = np.zeros((M,)*3, order="F")
        support_ = None
        rho_ = None
    comm.Bcast(ac_phased, root=0)

    return ac_phased, support_, rho_
