# Test that forward and adjoint transforms from both finufft & cufinufft
# are equivalent to the expected (pre-calculated) values.


from spinifel.sequential.orientation_matching import match 
from spinifel.slicing import gen_model_slices_batch
import skopi as skp
import h5py
import pickle
import numpy as np
import os
from spinifel import prep

N_slices = 10
test_data_dir = os.environ['test_data_dir']

from dataclasses import dataclass

@dataclass
class InputData:
    name: str
    slices_: np.array
    pixel_position_reciprocal: np.array
    known_quaternions: np.array
    volume: np.array
    ref_orientations: np.array
    n_pixels: int
    ac_support_size: int
    oversampling: int
    batch_size: int
    
    @property
    def ac_phased(self) -> np.array:
        ivol = np.square(np.abs(self.volume))
        ac_phased = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))).astype(np.float32)
        return ac_phased
        
def test_match(test_case):

    print(f'Test case: {test_case}')
    print(f'Test orientation matching N_slices: {N_slices}')
    
    # load input test
    if test_case == "2cex":
        # Input data from this test case uses the original spinifel input data
        # NOTE:
        # known_orientations is the correct orientations (3)
        # ref_orientations just take the correct ones + the next orientations as stored in
        # the file.
        test_data = h5py.File(os.path.join(test_data_dir,'2CEX','2CEX-10.h5'), 'r')
        n_pixels = 4 * 512 * 512
        ac_support_size = 151
        oversampling = 1
        batch_size = 10
        N_test_orientations = 10

        # h5py stores panels, det_x, det_y, dimension (4, 512, 512, 3)
        # need to move dimension to the first column (3, 4, 512, 512)
        pixel_position_reciprocal = np.moveaxis(
                test_data['pixel_position_reciprocal'][:], -1, 0)
        inp = InputData(test_case, test_data['intensities'][:N_slices],
                pixel_position_reciprocal,
                test_data['orientations'],
                test_data['volume'],
                test_data['orientations'][:N_test_orientations],
                n_pixels, 
                ac_support_size,
                oversampling,
                batch_size)

    slices_ = inp.slices_
    pixel_position_reciprocal = inp.pixel_position_reciprocal
    known_quaternions = inp.known_quaternions
    ac_phased = inp.ac_phased
    ref_orientations = inp.ref_orientations
    N_pixels = inp.n_pixels
    N_batch_size = inp.batch_size
    ac_support_size = inp.ac_support_size
    oversampling = inp.oversampling

    # calculate pixel distance to get max
    pixel_distance_reciprocal = prep.compute_pixel_distance(
        pixel_position_reciprocal)
    reciprocal_extent = pixel_distance_reciprocal.max()

    # flatten images
    slices_ = slices_.reshape((slices_.shape[0], N_pixels))

    # generate model slices
    model_slices_cpu = gen_model_slices_batch(ac_phased, ref_orientations, 
            pixel_position_reciprocal, reciprocal_extent, oversampling, 
            ac_support_size, N_pixels, batch_size=N_batch_size,
            override_forward_with='cpu')
    model_slices_gpu = gen_model_slices_batch(ac_phased, ref_orientations, 
            pixel_position_reciprocal, reciprocal_extent, oversampling, 
            ac_support_size, N_pixels, batch_size=N_batch_size,
            override_forward_with='gpu')
    
    print(f'Input data:')
    print(f'  test case: {inp.name}')
    print(f'  slices_: {slices_.shape} dtype: {slices_.dtype}')
    print(f'  pixel_position_reciprocal: {pixel_position_reciprocal.shape} dtype: {pixel_position_reciprocal.dtype}')
    print(f'  pixel_distance_reciprocal: {pixel_distance_reciprocal.shape} dtype: {pixel_distance_reciprocal.dtype}')
    print(f'  reciprocal_extent: {reciprocal_extent} dtype: {type(reciprocal_extent)}')
    
    print(f'  ac_phased: {ac_phased.shape} dtype: {ac_phased.dtype}')
    print(f'  ref_orientations: {ref_orientations.shape} dtype: {ref_orientations.dtype}')
    print(f'  N_pixels: {N_pixels}')
    print(f'  N_batch_size: {N_batch_size}')
    print(f'  oversampling: {oversampling}')
    print(f'  ac_support_size: {ac_support_size}')
    print(f'Compare Forward cpu and gpu:')
    print(f'  dtype cpu:{model_slices_cpu.dtype} gpu:{model_slices_gpu.dtype}')
    print(f'  shape cpu:{model_slices_cpu.shape} gpu:{model_slices_gpu.shape}')
    print(f'  equal?', np.array_equal(model_slices_cpu, model_slices_gpu))
    cc = np.corrcoef(model_slices_cpu.flatten(), model_slices_gpu.flatten())[0,1]
    print(f'  cc:', cc)
    deltas = model_slices_cpu.flatten()-model_slices_gpu.flatten()
    r_value = np.sum(np.abs(deltas))/np.sum(np.abs(model_slices_cpu[:]))
    print(f'  r:', r_value)
    print(f'  r is sum(cpu)/sum(deltas(cpu, gpu))')
    assert cc > 1-1e-9
    assert r_value < 1e-6



if __name__ == "__main__":
    test_match("2cex")
