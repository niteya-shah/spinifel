import numpy as np
from scipy.ndimage import gaussian_filter

from spinifel import parms, image


# Convention:
#   In this module, trailing underscores are used to refer to numpy
# arrays that have been ifftshifted.
# For unshifted arrays, the FFT/IFFT are defined as:
#   f -> np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f))) / Mtot
#   f -> np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f))) * Mtot
# For shifted arrays, the FFT/IFFT are thus as:
#   f_ -> np.fft.fftn(f_) / Mtot
#   f_ -> np.fft.ifftn(f_) * Mtot


def create_support_(ac_, M, Mquat):
    sl = slice(Mquat, -Mquat)
    square_support = np.zeros((M, M, M), dtype=np.bool_)
    square_support[sl, sl, sl] = 1
    square_support_ = np.fft.ifftshift(square_support)
    image.show_volume(square_support, Mquat, "square_support_0.png")

    thresh_support_ = ac_ > 1e-2 * ac_.max()
    image.show_volume(np.fft.fftshift(thresh_support_), Mquat, "thresh_support_0.png")

    return np.logical_and(square_support_, thresh_support_)


def phase(ac):
    Mquat = parms.Mquat
    M = 4*Mquat + 1
    Mtot = M**3

    ac_filt = gaussian_filter(np.maximum(ac.real, 0), mode='constant',
                              sigma=1, truncate=2)
    image.show_volume(ac_filt, Mquat, "autocorrelation_filtered_0.png")
    ac_filt_ = np.fft.ifftshift(ac_filt)

    fft = lambda f_: np.fft.fftn(f_) / Mtot
    ifft = lambda f_: np.fft.ifftn(f_) * Mtot

    intensities_ = np.abs(fft(ac_filt_))
    image.show_volume(np.fft.fftshift(intensities_), Mquat, "intensities_0.png")

    amplitudes_ = np.sqrt(intensities_)
    image.show_volume(np.fft.fftshift(amplitudes_), Mquat, "amplitudes_0.png")

    masked_amp_ = amplitudes_.copy()
    masked_amp_[0, 0, 0] = np.nan

    support_ = create_support_(ac_filt_, M, Mquat)
    image.show_volume(np.fft.fftshift(support_), Mquat, "support_0.png")

    rho_ = support_ * np.random.rand(*support_.shape)
    image.show_volume(np.fft.fftshift(rho_), Mquat, "rho_0.png")
