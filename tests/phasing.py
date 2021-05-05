import os
import numpy as np
import pytest
import time

from spinifel.sequential.phasing import center_of_mass, recenter, create_support, ER_loop, HIO_loop, ER, HIO, step_phase, shrink_wrap

test_data_dir = os.environ['test_data_dir']

def phase(generation, ac, support_=None, rho_=None):
    Mquat = 37
    M = 4*Mquat + 1
    Mtot = M**3

    ac_filt = gaussian_filter(np.maximum(ac.real, 0), mode='constant',
                              sigma=1, truncate=2)
    ac_filt_ = np.fft.ifftshift(ac_filt)

    intensities_ = np.abs(np.fft.fftn(ac_filt_))

    amplitudes_ = np.sqrt(intensities_)

    amp_mask_ = np.ones((M, M, M), dtype=np.bool_)
    amp_mask_[0, 0, 0] = 0  # Mask out central peak

    if support_ is None:
        support_ = create_support_(ac_filt_, M, Mquat, generation)

    if rho_ is None:
        rho_ = support_ * np.random.rand(*support_.shape)

    rho_max = np.infty

    N_phase_loops = 10
    nER = 3
    nHIO = 2
    
    for i in range(N_phase_loops):
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        HIO_loop(nHIO, 0.9, rho_, amplitudes_, amp_mask_, support_, rho_max)
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        shrink_wrap(0.1, 1, rho_, support_)
    ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)

    recenter(rho_, support_, M)

    intensities_phased_ = np.abs(np.fft.fftn(rho_))**2

    ac_phased_ = np.abs(np.fft.ifftn(intensities_phased_))
    ac_phased = np.fft.fftshift(ac_phased_)

    return ac_phased, support_, rho_


class TestPhase(object):
    """Test phasing using the precomputed ground-truth autocorrelation and ground-truth density"""

    @classmethod
    def setup_class(cls):
        # read in precomputed data
        ref = np.load(os.path.join(test_data_dir, 'reference.npz')
        density_ref = ref['density']
        support_ref = ref['support']
        volume_ref = np.abs(np.fft.fftn(density_ref))**2
        ac_ref = np.fft.fftshift(np.abs(np.fft.ifftn(volume_ref)))
        support_ref_ = np.fft.ifftshift(support_ref)
        Mquat = int((ac_ref.shape[0]-1)/4) # (149-1)/4=37
        M = ac_ref.shape[0] # 149
        Mtot = M**3
        cls.ac_ref = ac_ref
        cls.density_ref = density_ref
        cls.support_ref_ = support_ref_

    def test_phasing(self):
        ac_phased, support_, rho_ = phase(generation=0, ac=self.ac_ref, support_=self.support_ref_, rho_=None)
        density_calc = np.fft.ifftshift(rho_)
        for i in range(3):
            for j in range(3):
                if i >= j:
                    proj_ref = np.sum(self.density_ref, axis=i)
                    proj_calc = np.sum(density_calc, axis=j)
                    proj_cc = np.corrcoef(proj_ref.flatten(), proj_calc.flatten())[0,1]
                    if i==j: # corresponding slices match
                        assert proj_cc > 0.9
                    elif i!=2 and j!=2: # additional matches due to this protein's symmetry
                        assert proj_cc > 0.9
                    else:
                        assert proj_cc < 0.9

