import numpy as np
from scipy.ndimage import gaussian_filter

from spinifel import parms, image


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


def center_of_mass(rho_, hkl_, M):
    rho_ = np.abs(rho_)
    num = (rho_ * hkl_).sum(axis=(1, 2, 3))
    den = rho_.sum()
    return np.round(num/den * M/2)


def recenter(rho_, support_, M):
    ls = np.linspace(-1, 1, M+1)
    ls = (ls[:-1] + ls[1:])/2

    hkl_list = np.meshgrid(ls, ls, ls, indexing='ij')
    hkl_ = np.stack([np.fft.ifftshift(coord) for coord in hkl_list])
    vect = center_of_mass(rho_, hkl_, M)

    for i in range(3):
        shift = int(vect[i])
        rho_[:] = np.roll(rho_, -shift, i)
        support_[:] = np.roll(support_, -shift, i)


def create_support_(ac_, M, Mquat, generation):
    sl = slice(Mquat, -Mquat)
    square_support = np.zeros((M, M, M), dtype=np.bool_)
    square_support[sl, sl, sl] = 1
    square_support_ = np.fft.ifftshift(square_support)
    image.show_volume(square_support, Mquat, f"square_support_{generation}.png")

    thresh_support_ = ac_ > 1e-2 * ac_.max()
    image.show_volume(np.fft.fftshift(thresh_support_), Mquat, f"thresh_support_{generation}.png")

    return np.logical_and(square_support_, thresh_support_)


def ER_loop(n_loops, rho_, amplitudes_, amp_mask_, support_, rho_max):
    for k in range(n_loops):
        ER(rho_, amplitudes_, amp_mask_, support_, rho_max)


def HIO_loop(n_loops, beta, rho_, amplitudes_, amp_mask_, support_, rho_max):
    for k in range(n_loops):
        HIO(beta, rho_, amplitudes_, amp_mask_, support_, rho_max)


def ER(rho_, amplitudes_, amp_mask_, support_, rho_max):
    rho_mod_, support_star_ = step_phase(rho_, amplitudes_, amp_mask_, support_)
    rho_[:] = np.where(support_star_, rho_mod_, 0)
    i_overmax = rho_mod_ > rho_max
    rho_[i_overmax] = rho_max


def HIO(beta, rho_, amplitudes_, amp_mask_, support_, rho_max):
    rho_mod_, support_star_ = step_phase(rho_, amplitudes_, amp_mask_, support_)
    rho_[:] = np.where(support_star_, rho_mod_, rho_-beta*rho_mod_)
    i_overmax = rho_mod_ > rho_max
    rho_[i_overmax] += 2*beta*rho_mod_[i_overmax] - rho_max


def step_phase(rho_, amplitudes_, amp_mask_, support_):
    rho_hat_ = np.fft.fftn(rho_)
    phases_ = np.angle(rho_hat_)
    rho_hat_mod_ = np.where(
        amp_mask_,
        amplitudes_ * np.exp(1j*phases_),
        rho_hat_)
    rho_mod_ = np.fft.ifftn(rho_hat_mod_).real
    support_star_ = np.logical_and(support_, rho_mod_>0)
    return rho_mod_, support_star_


def shrink_wrap(cutoff, sigma, rho_, support_):
    rho_abs_ = np.absolute(rho_)
    # By using 'wrap', we don't need to fftshift it back and forth
    rho_gauss_ = gaussian_filter(
        rho_abs_, mode='wrap', sigma=sigma, truncate=2)
    support_[:] = rho_gauss_ > rho_abs_.max() * cutoff


def phase(generation, ac, support_=None, rho_=None):
    Mquat = parms.Mquat
    M = 4*Mquat + 1
    Mtot = M**3

    ac_filt = gaussian_filter(np.maximum(ac.real, 0), mode='constant',
                              sigma=1, truncate=2)
    image.show_volume(ac_filt, Mquat, f"autocorrelation_filtered_{generation}.png")
    ac_filt_ = np.fft.ifftshift(ac_filt)

    intensities_ = np.abs(np.fft.fftn(ac_filt_))
    image.show_volume(np.fft.fftshift(intensities_), Mquat, f"intensities_{generation}.png")

    amplitudes_ = np.sqrt(intensities_)
    image.show_volume(np.fft.fftshift(amplitudes_), Mquat, f"amplitudes_{generation}.png")

    amp_mask_ = np.ones((M, M, M), dtype=np.bool_)
    amp_mask_[0, 0, 0] = 0  # Mask out central peak
    image.show_volume(np.fft.fftshift(amp_mask_), Mquat, f"amp_mask_{generation}.png")

    if support_ is None:
        support_ = create_support_(ac_filt_, M, Mquat, generation)
    image.show_volume(np.fft.fftshift(support_), Mquat, f"support_{generation}.png")

    if rho_ is None:
        rho_ = support_ * np.random.rand(*support_.shape)
    image.show_volume(np.fft.fftshift(rho_), Mquat, f"rho_{generation}.png")

    rho_max = np.infty

    nER = parms.nER
    nHIO = parms.nHIO

    for i in range(parms.N_phase_loops):
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        HIO_loop(nHIO, 0.3, rho_, amplitudes_, amp_mask_, support_, rho_max)
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        shrink_wrap(5e-2, 1, rho_, support_)
    ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)

    recenter(rho_, support_, M)

    image.show_volume(np.fft.fftshift(rho_), Mquat, f"rho_phased_{generation}.png")

    intensities_phased_ = np.abs(np.fft.fftn(rho_))**2
    image.show_volume(np.fft.fftshift(intensities_phased_), Mquat, f"intensities_phased_{generation}.png")

    ac_phased_ = np.abs(np.fft.ifftn(intensities_phased_))
    ac_phased = np.fft.fftshift(ac_phased_)
    image.show_volume(ac_phased, Mquat, f"autocorrelation_phased_{generation}.png")

    ac_phased = ac_phased.astype(np.float32)

    return ac_phased, support_, rho_
