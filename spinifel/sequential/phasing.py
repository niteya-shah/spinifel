import numpy as np
from scipy.ndimage import gaussian_filter

from spinifel import parms, image


def phase(ac):
    Mquat = parms.Mquat
    M = 4*Mquat + 1
    Mtot = M**3

    ac_filt = gaussian_filter(np.maximum(ac.real, 0), mode='constant',
                              sigma=1, truncate=2)
    image.show_volume(ac_filt, Mquat, "autocorrelation_filtered_0.png")

    fft = lambda f: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f))) / Mtot
    ifft = lambda f: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f))) * Mtot

    intensities = np.abs(fft(ac_filt))
    image.show_volume(intensities, Mquat, "intensities_0.png")

    amplitudes = np.sqrt(intensities)
    image.show_volume(amplitudes, Mquat, "amplitudes_0.png")
