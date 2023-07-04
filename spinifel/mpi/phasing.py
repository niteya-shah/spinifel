import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skew
import matplotlib.pyplot as plt

from mpi4py import MPI

from spinifel import settings, contexts, utils
from spinifel.sequential.phasing import phase as sequential_phase

# import omnitrace

# @omnitrace.profile()
def phase(generation, ac, support_=None, rho_=None):
    """Phase retrieval by Rank0 and broadcast to all ranks."""

    comm = contexts.comm_compute
    logger = utils.Logger(True, settings, myrank=comm.rank)

    weight = 0.5 + comm.rank / comm.size  # shrinkwrap weight range: 0.5 to 1.5
    method = "std"
    ac_phased, support_, rho_ = sequential_phase(
        generation, ac, support_, rho_, method, weight
    )

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

    # fit_y = Gauss(xdata, fit_A, fit_B, fit_C)
    # MAE = np.sum(np.abs(fit_y - ydata))/len(xdata) # maximum abs. error
    skewness = skew(ydata)
    # Rank0 gathers peak height, width, and skew from all ranks
    summary = comm.gather((comm.rank, np.max(ydata), fit_C, skewness), root=0)
    if comm.rank == 0:
        ranks, heights, widths, skews = [np.array(el) for el in zip(*summary)]
        iref = np.argmin(skews)
        ref_rank = ranks[iref]
        logger.log(f"Skewness summary:", level=1)
        logger.log(f"heights: {heights}", level=1)
        logger.log(f"widths: {widths}", level=1)
        logger.log(f"skews: {skews}", level=1)
        logger.log(
            f"Keeping result from rank {ref_rank}: skew={skews[iref]:.2f}", level=1
        )
    else:
        ref_rank = -1
    ref_rank = comm.bcast(ref_rank, root=0)

    comm.Bcast(ac_phased, root=ref_rank)
    comm.Bcast(support_, root=ref_rank)
    comm.Bcast(rho_, root=ref_rank)

    return ac_phased, support_, rho_
