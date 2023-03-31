import os
import numpy as np
import PyNVTX as nvtx

from spinifel import SpinifelSettings, settings, image, utils

# only for timing
import time
# settings = SpinifelSettings()
logger = utils.Logger(True, settings)

xp = np
if settings.use_cupy:
    logger.log(f"Using CuPy for FFTs.", level=1)
    import cupy as xp
    from cupyx.scipy.ndimage import gaussian_filter
else:
    logger.log(f"Using NumPy for FFTs.", level=1)
    from scipy.ndimage import gaussian_filter

if settings.use_fftx:
    if settings.verbose:
        print(f"Using FFTX for FFTs.")
        import fftx as fftxp
        # fftx_options_cuda = {'cuda' : True}
        # fftx_options_nocuda = {'cuda' : False}

if settings.use_single_prec:
    f_type = xp.float32
    c_type = xp.complex64
else:
    f_type = xp.float64
    c_type = xp.complex128

# Convention:
#   In this module, trailing underscores are used to refer to numpy
# arrays that have been ifftshifted.
# For unshifted arrays, the FFT/IFFT are defined as:
#   f -> np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))
#   f -> np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f)))
# For shifted arrays, the FFT/IFFT are thus as:
#   f_ -> np.fft.fftn(f_)
#   f_ -> np.fft.ifftn(f_)
# To be compatible with the conventions used in the AC solver,
# we would also have to scale by Mtot:
#   f -> np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f))) / Mtot
#   f -> np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f))) * Mtot
# but this won't be done here at each iteration.
# Please mind the difference when comparing results.


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def center_of_mass(rho_, hkl_, M):
    """
    Compute the object's center of mass.

    :param rho_: electron density (fftshifted)
    :param hkl_: coordinates
    :param M: cubic length of electron density volume
    :return vect: vector center of mass in units of pixels
    """
    rho_ = np.abs(rho_)
    num = (rho_ * hkl_).sum(axis=(1, 2, 3))
    den = rho_.sum()
    return np.round(num / den * M / 2)


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def recenter(rho_, support_, M):
    """
    Shift center of the electron density and support to origin.

    :param rho_: electron density (fftshifted)
    :param support_: object's support
    :param M: cubic length of electron density volume
    """
    ls = np.linspace(-1, 1, M + 1)
    ls = (ls[:-1] + ls[1:]) / 2

    hkl_list = np.meshgrid(ls, ls, ls, indexing="ij")
    hkl_ = np.stack([np.fft.ifftshift(coord) for coord in hkl_list])
    vect = center_of_mass(rho_, hkl_, M)

    for i in range(3):
        shift = int(vect[i])
        rho_[:] = np.roll(rho_, -shift, i)
        support_[:] = np.roll(support_, -shift, i)


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def create_support_(ac_, M, Mquat, generation):
    """
    Generate a support based on the region of high ACF signal (thresh_support_)
    inside the central quarter region of the full ACF volume (square_support_).

    :param ac_: autocorrelation volume, fftshifted
    :param M: cubic length of autocorrelation volume
    :param Mquat: cubic length of region of interest
    :param generation: current iteration
    """
    sl = slice(Mquat, -Mquat)
    square_support = xp.zeros((M, M, M), dtype=xp.bool_)
    square_support[sl, sl, sl] = 1
    square_support_ = xp.fft.ifftshift(square_support)
    # image.show_volume(square_support, Mquat, f"square_support_{generation}.png")

    thresh_support_ = ac_ > 1e-2 * ac_.max()
    # image.show_volume(np.fft.fftshift(thresh_support_), Mquat, f"thresh_support_{generation}.png")

    return xp.logical_and(square_support_, thresh_support_)


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def ER_loop(n_loops, rho_, amplitudes_, amp_mask_, support_, rho_max):
    """
    Perform n_loops of Error Reduction (ER) opertation.
    """
    for k in range(n_loops):
        ER(rho_, amplitudes_, amp_mask_, support_, rho_max)


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def HIO_loop(n_loops, beta, rho_, amplitudes_, amp_mask_, support_, rho_max):
    """
    Perform n_loops of Hybrid-Input-Output (HIO) opertation.
    """
    for k in range(n_loops):
        HIO(beta, rho_, amplitudes_, amp_mask_, support_, rho_max)


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def ER(rho_, amplitudes_, amp_mask_, support_, rho_max):
    """
    Perform Error Reduction (ER) operation by updating the amplitudes from the current electron density estimtate
    with those computed from the autocorrelation, and enforcing electron density to be positive (real space constraint).

    :param rho_: current electron density estimate
    :param amplitudes_: amplitudes computed from the autocorrelation
    :param amp_mask_: amplitude mask
    :param support_: binary mask for object's support
    :param rho_mask: maximum permitted electron density value
    """
    rho_mod_, support_star_ = step_phase(rho_, amplitudes_, amp_mask_, support_)
    rho_[:] = xp.where(support_star_, rho_mod_, 0)
    i_overmax = rho_mod_ > rho_max
    rho_[i_overmax] = rho_max


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def HIO(beta, rho_, amplitudes_, amp_mask_, support_, rho_max):
    """
    Perform Hybrid-Input-Output (HIO) operation by updating the amplitudes from the current electron density estimtate
    with those computed from the autocorrelation, and using negative feedback in Fourier space in order to progressively
    force the solution to conform to the Fourier domain constraints (support).

    :param beta: feedback constant
    :param rho_: electron density estimate
    :param amplitudes_: amplitudes computed from the autocorrelation
    :param amp_mask_: amplitude mask
    :param support_: binary mask for object's support
    :param rho_mask: maximum permitted electron density value
    """
    rho_mod_, support_star_ = step_phase(rho_, amplitudes_, amp_mask_, support_)
    rho_[:] = xp.where(support_star_, rho_mod_, rho_ - beta * rho_mod_)
    i_overmax = rho_mod_ > rho_max
    rho_[i_overmax] += 2 * beta * rho_mod_[i_overmax] - rho_max


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def step_phase(rho_, amplitudes_, amp_mask_, support_):
    """
    Replace the amplitudes computed from the electron density estimate with those computed from
    the autocorrelation function, except for the amplitude of the central/max peak of the Fourier domain.
    Then recalculate the estimated electron density and update the support, with positivity of
    the density enforced for the latter.

    :param rho_: electron density estimate
    :param amplitudes_: amplitudes computed from the autocorrelation
    :param amp_mask_: amplitude mask
    :param support_: binary mask for object's support
    :return rho_mod_: updated density estimate
    :return support_star_: updated support
    """
    start_time = time.time()
    rho_hat_ = xp.fft.fftn(rho_)
    phases_ = xp.angle(rho_hat_)
    rho_hat_mod_ = xp.where(amp_mask_, amplitudes_ * xp.exp(1j * phases_), rho_hat_)
    rho_mod_ = xp.fft.ifftn(rho_hat_mod_).real
    end_time = time.time()
    xp_time = end_time - start_time
    if settings.use_fftx:
        start_time = time.time()
        # rho_complex = rho_.astype(dtype=xp.complex128, order='C')
        # rho_mod_fftx = fftxp.kernels.step_phase_kernel(xp, rho_complex, amp_mask_, amplitudes_)
        # REUSE phases_ to store rho_mod_fftx.
        # Need an array of shape=(M, M, M) dtype=float64 C=True CuPy=True.
        # rho_mod_fftx = fftxp.convo.stepphase(rho_, amplitudes_)
        phases_ = fftxp.convo.stepphase(rho_, amplitudes_, phases_) # REUSE phases_ for rho_mod_fftx
        end_time = time.time()
        fftxp_time = end_time - start_time
        # fftxp.utils.print_diff(xp, rho_mod_, rho_mod_fftx, "step_phase rho_mod_")
        fftxp.utils.print_diff(xp, rho_mod_, phases_, "step_phase rho_mod_") # REUSE phases_ for rho_mod_fftx
        print(f"FULL TIME step_phase: xp {xp_time} fftxp {fftxp_time}")
        fftxp.utils.print_array_info(xp, rho_, "DATA rho_")
        fftxp.utils.print_array_info(xp, rho_mod_, "DATA rho_mod_")
        # fftxp.utils.print_array_info(xp, rho_mod_fftx, "DATA rho_mod_fftx")
        fftxp.utils.print_array_info(xp, phases_, "DATA rho_mod_fftx") # REUSE phases_ for rho_mod_fftx

    support_star_ = xp.logical_and(support_, rho_mod_>0)
    return rho_mod_, support_star_


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def shrink_wrap(sigma, rho_, support_, method=None, weight=1.0, cutoff=0.05):
    """
    Perform shrinkwrap operation to update the support for convergence.

    :param sigma: Gaussian standard deviation to low-pass filter density with
    :param rho_: electron density estimate
    :param support_: object support
    :param method: {'max', 'std'}, default: std
    kwargs:
    :param cutoff: method='max', threshold as a fraction of maximum density value
    :param weight: method='std', threshold as standard deviation of density times a weight factor
    """
    rho_abs_ = xp.absolute(rho_)
    # By using 'wrap', we don't need to fftshift it back and forth
    rho_gauss_ = gaussian_filter(rho_abs_, mode="wrap", sigma=sigma, truncate=2)
    if method == None:
        method = "std"
    if method == "std":
        threshold = xp.std(rho_gauss_) * weight
    elif method == "max":
        threshold = rho_abs_.max() * cutoff * weight
    else:
        raise ValueError(f"Invalid method: {method}. Options are 'std' or 'max'.")
    support_[:] = rho_gauss_ > threshold


@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def phase(
    generation, ac, support_=None, rho_=None, method=None, weight=1.0, group_idx=0
):
    """
    Solve phase retrieval from the autocorrelation of the current electron density estimate
    by performing cycles of ER/HIO/shrinkwrap combination.
    Note that this function currently is composed of three components:
    (1) convert ac to amplitude,
    (2) perform phase retrieval,
    (3) convert rho_ to ac_phased.
    We might revisit it to break it down to three modules.

    :param generation: current iteration of M-TIP loop
    :param ac: autocorrelation of the current electron density estimate
    :param support_: initial object support
    :param rho_: initial electron density estimate
    :return ac_phased: updated autocorrelation estimate
    :return support_: updated support estimate
    :return rho_: updated density estimate
    """

    Mquat = settings.Mquat
    M = 4 * Mquat + 1
    Mtot = M**3

    ac = xp.array(ac)
    ac_filt = gaussian_filter(
        xp.maximum(ac.real, 0), mode="constant", sigma=1, truncate=2
    )
    ac_filt_ = xp.fft.ifftshift(ac_filt)

    start_time = time.time()
    intensities_ = xp.abs(xp.fft.fftn(ac_filt_))
    end_time = time.time()
    xp_time = end_time - start_time
    if settings.use_fftx:
        print(f"ORDER fftn ac_filt_ is {ac_filt_.flags.c_contiguous}")
        start_time = time.time()
        ac_filt_complex = ac_filt_.astype(xp.complex128, order='C')
        intensities_fftx = xp.abs(fftxp.fft.fftn(ac_filt_complex))
        end_time = time.time()
        fftxp_time = end_time - start_time
        fftxp.utils.print_diff(xp, intensities_, intensities_fftx, "intensities_")
        print(f"FULL TIME intensities_: xp {xp_time} fftxp {fftxp_time}")
        fftxp.utils.print_array_info(xp, ac_filt_, "DATA ac_filt_")

    #image.show_volume(xp.fft.fftshift(intensities_), Mquat, f"intensities_{generation}.png")
    amplitudes_ = xp.sqrt(intensities_)

    amp_mask_ = xp.ones((M, M, M), dtype=xp.bool_)
    amp_mask_[0, 0, 0] = 0  # Mask out central peak

    if support_ is None:
        support_ = create_support_(ac_filt_, M, Mquat, generation)
    support_ = xp.array(support_)

    if rho_ is None:
        rho_ = support_ * xp.random.rand(*support_.shape)
    rho_ = xp.array(rho_)

    rho_max = xp.infty

    nER = settings.nER
    nHIO = settings.nHIO

    for i in range(settings.N_phase_loops):
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        HIO_loop(nHIO, settings.beta, rho_, amplitudes_, amp_mask_, support_, rho_max)
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        shrink_wrap(1, rho_, support_, method=method, weight=weight)
    ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)

    if settings.use_cupy:
        logger.log(f"Converting CuPy arrays to NumPy arrays.", level=1)
        rho_ = xp.asnumpy(rho_)
        amplitudes_ = xp.asnumpy(amplitudes_)
        amp_mask_ = xp.asnumpy(amp_mask_)
        support_ = xp.asnumpy(support_)

    recenter(rho_, support_, M)

    image.show_volume(
        np.fft.fftshift(rho_), Mquat, f"rho_phased_{generation}_{group_idx}.png"
    )

    start_time = time.time()
    intensities_phased_ = np.abs(np.fft.fftn(rho_))**2
    end_time = time.time()
    np_time = end_time - start_time
    if settings.use_fftx:
        print(f"ORDER fftn rho_ is {rho_.flags.c_contiguous}")
        start_time = time.time()
        rho_complex = rho_.astype(np.complex128, order='C')
        intensities_phased_fftx = np.abs(fftxp.fft.fftn(rho_complex))**2
        end_time = time.time()
        fftxp_time = end_time - start_time
        fftxp.utils.print_diff(np, intensities_phased_, intensities_phased_fftx,
                               "intensities_phased_")
        print(f"FULL TIME intensities_phased_: np {np_time} fftxp {fftxp_time}")
        fftxp.utils.print_array_info(np, rho_, "DATA for intensities rho_")

    image.show_volume(
        np.fft.fftshift(intensities_phased_),
        Mquat,
        f"intensities_phased_{generation}_{group_idx}.png",
    )
    start_time = time.time()
    ac_phased_ = np.abs(np.fft.ifftn(intensities_phased_))
    end_time = time.time()
    np_time = end_time - start_time
    if settings.use_fftx:
        print(f"ORDER ifftn intensities_phased_ is {intensities_phased_.flags.c_contiguous}")
        start_time = time.time()
        intensities_phased_complex = intensities_phased_.astype(np.complex128,
                                                                order='C')
        ac_phased_fftx = np.abs(fftxp.fft.ifftn(intensities_phased_complex))
        end_time = time.time()
        fftxp_time = end_time - start_time
        print(f"ac_phased_ norms original {np.max(np.absolute(ac_phased_))} FFTX {np.max(np.absolute(ac_phased_fftx))}")
        fftxp.utils.print_diff(np, ac_phased_, ac_phased_fftx, "ac_phased_")
        print(f"FULL TIME ac_phased_: np {np_time} fftxp {fftxp_time}")
        fftxp.utils.print_array_info(np, intensities_phased_, "DATA intensities_phased_")

    ac_phased = np.fft.fftshift(ac_phased_)
    # image.show_volume(ac_phased, Mquat, f"autocorrelation_phased_{generation}.png")

    ac_phased = ac_phased.astype(f_type)
    return ac_phased, support_, rho_
