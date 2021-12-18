import os
import numpy  as np
import PyNVTX as nvtx

from spinifel import SpinifelSettings, settings, image

# settings = SpinifelSettings()

xp = np
if settings.use_cupy:
    if settings.verbose:
        print(f"Using CuPy for FFTs.")
    import cupy as xp
    from cupyx.scipy.ndimage import gaussian_filter
else:
    if settings.verbose:
        print(f"Using NumPy for FFTs.")
    from scipy.ndimage import gaussian_filter


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
    rho_ = xp.abs(rho_)
    num = (rho_ * hkl_).sum(axis=(1, 2, 3))
    den = rho_.sum()
    return xp.round(num/den * M/2)



@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def recenter(rho_, support_, M):
    """
    Shift center of the electron density and support to origin.
    
    :param rho_: electron density (fftshifted)
    :param support_: object's support
    :param M: cubic length of electron density volume
    """
    ls = xp.linspace(-1, 1, M+1)
    ls = (ls[:-1] + ls[1:])/2

    hkl_list = xp.meshgrid(ls, ls, ls, indexing='ij')
    hkl_ = xp.stack([xp.fft.ifftshift(coord) for coord in hkl_list])
    vect = center_of_mass(rho_, hkl_, M)

    for i in range(3):
        shift = int(vect[i])
        rho_[:] = xp.roll(rho_, -shift, i)
        support_[:] = xp.roll(support_, -shift, i)



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

    thresh_support_ = ac_ > 1e-2 * ac_.max()

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
    rho_[:] = xp.where(support_star_, rho_mod_, rho_-beta*rho_mod_)
    i_overmax = rho_mod_ > rho_max
    rho_[i_overmax] += 2*beta*rho_mod_[i_overmax] - rho_max



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
    rho_hat_ = xp.fft.fftn(rho_)
    phases_ = xp.angle(rho_hat_)
    rho_hat_mod_ = xp.where(
        amp_mask_,
        amplitudes_ * xp.exp(1j*phases_),
        rho_hat_)
    rho_mod_ = xp.fft.ifftn(rho_hat_mod_).real
    support_star_ = xp.logical_and(support_, rho_mod_>0)
    return rho_mod_, support_star_



@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def shrink_wrap(cutoff, sigma, rho_, support_):
    """
    Perform shrinkwrap operation to update the support for convergence.

    :param cutoff: threshold as a fraction of maximum density value
    :param sigma: Gaussian standard deviation to low-pass filter density with
    :param rho_: electron density estimate
    :param support_: object support
    """
    rho_abs_ = xp.absolute(rho_)
    # By using 'wrap', we don't need to fftshift it back and forth
    rho_gauss_ = gaussian_filter(
        rho_abs_, mode='wrap', sigma=sigma, truncate=2)
    support_[:] = rho_gauss_ > rho_abs_.max() * cutoff



@nvtx.annotate("sequential/phasing.py", is_prefix=True)
def phase(generation, ac, support_=None, rho_=None):
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
    M = 4*Mquat + 1
    Mtot = M**3

    ac = xp.array(ac)
    ac_filt = gaussian_filter(xp.maximum(ac.real, 0), mode='constant',
                              sigma=1, truncate=2)
    #image.show_volume(ac_filt, Mquat, f"autocorrelation_filtered_{generation}.png")
    ac_filt_ = xp.fft.ifftshift(ac_filt)

    intensities_ = xp.abs(xp.fft.fftn(ac_filt_))
    #image.show_volume(xp.fft.fftshift(intensities_), Mquat, f"intensities_{generation}.png")

    amplitudes_ = xp.sqrt(intensities_)
    #image.show_volume(xp.fft.fftshift(amplitudes_), Mquat, f"amplitudes_{generation}.png")

    amp_mask_ = xp.ones((M, M, M), dtype=xp.bool_)
    amp_mask_[0, 0, 0] = 0  # Mask out central peak
    #image.show_volume(xp.fft.fftshift(amp_mask_), Mquat, f"amp_mask_{generation}.png")

    if support_ is None:
        support_ = create_support_(ac_filt_, M, Mquat, generation)
    #image.show_volume(xp.fft.fftshift(support_), Mquat, f"support_{generation}.png")
    support_ = xp.array(support_)

    if rho_ is None:
        rho_ = support_ * xp.random.rand(*support_.shape)
    #image.show_volume(xp.fft.fftshift(rho_), Mquat, f"rho_{generation}.png")
    rho_ = xp.array(rho_)
    
    rho_max = xp.infty

    nER = settings.nER
    nHIO = settings.nHIO

    for i in range(settings.N_phase_loops):
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        HIO_loop(nHIO, 0.9, rho_, amplitudes_, amp_mask_, support_, rho_max)
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        shrink_wrap(0.1, 1, rho_, support_)
    ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)

    if settings.use_cupy:
        if settings.verbose:
            print(f"Converting CuPy arrays to NumPy arrays.")    
        rho_ = xp.asnumpy(rho_)
        amplitudes_ = xp.asnumpy(amplitudes_)
        amp_mask_ = xp.asnumpy(amp_mask_)
        support_ = xp.asnumpy(support_)

    recenter(rho_, support_, M)

    image.show_volume(np.fft.fftshift(rho_), Mquat, f"rho_phased_{generation}.png")

    intensities_phased_ = np.abs(np.fft.fftn(rho_))**2
    image.show_volume(np.fft.fftshift(intensities_phased_), Mquat, f"intensities_phased_{generation}.png")

    ac_phased_ = np.abs(np.fft.ifftn(intensities_phased_))
    ac_phased = np.fft.fftshift(ac_phased_)
    #image.show_volume(ac_phased, Mquat, f"autocorrelation_phased_{generation}.png")

    ac_phased = ac_phased.astype(np.float32)

    return ac_phased, support_, rho_

def phase_cmtip(generation, ac, support_=None, rho_=None):
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

    Mquat = int((ac.shape[0]-1)/4) ##
    M = ac.shape[0] ##
    Mtot = M**3

    ac = xp.array(ac)
    ac_filt = gaussian_filter(xp.maximum(ac.real, 0), mode='constant',
                              sigma=1, truncate=2)
    ac_filt_ = xp.fft.ifftshift(ac_filt)

    intensities_ = xp.abs(xp.fft.fftn(ac_filt_))

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
        shrink_wrap(settings.cutoff, 1, rho_, support_)
    ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)

    if settings.use_cupy:
        if settings.verbose:
            print(f"Converting CuPy arrays to NumPy arrays.")    
        rho_ = xp.asnumpy(rho_)
        amplitudes_ = xp.asnumpy(amplitudes_)
        amp_mask_ = xp.asnumpy(amp_mask_)
        support_ = xp.asnumpy(support_)

    recenter(rho_, support_, M)

    intensities_phased_ = xp.abs(xp.fft.fftn(rho_))**2

    ac_phased_ = xp.abs(xp.fft.ifftn(intensities_phased_))
    ac_phased = xp.fft.fftshift(ac_phased_)

    return ac_phased, support_, rho_
