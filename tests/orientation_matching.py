# Test match() function for orientation matching in mpi module
# By giving 10,000 reference orientations (100 matched correctly with
# the selected 100 slides and the calculated ac (Autocorrelation), 
# orientation matching module should give correct quaternions.
# 


from spinifel.sequential.orientation_matching import slicing_and_match as match
import skopi as skp
import h5py
import pickle
import numpy as np
import os
from spinifel import prep

test_data_dir = os.environ.get('test_data_dir', '')
if not test_data_dir:
    print(f'test_data_dir not given. exit')
    exit()

from dataclasses import dataclass

@dataclass
class InputData:
    name: str
    slices_: np.array
    pixel_position_reciprocal: np.array
    pixel_index_map: np.array
    known_quaternions: np.array
    volume: np.array
    ref_orientations: np.array
    n_pixels: int
    ac_support_size: int
    oversampling: int
    
    @property
    def ac_phased(self) -> np.array:
        ivol = np.square(np.abs(self.volume))
        ac_phased = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))).astype(np.float32)
        return ac_phased

    def get_pixel_position_reciprocal(self):
        return self.pixel_position_reciprocal.astype(np.float32)
        


def test_match(test_case):
    print(f'Test case: {test_case}')
    
    # Load input test data. Note that known_orientations is the 
    # correct orientations as stored in hdf5 files
    if test_case == "3iyf":
        test_data = h5py.File(os.path.join(test_data_dir, '3IYF', '3iyf_sim_10k.h5'), 'r')
    elif test_case == "2cex":
        test_data = h5py.File(os.path.join(test_data_dir, '2CEX','2cex_sim_10k.h5'), 'r')

    # Shared spinifel parameters
    n_pixels = 1 * 128 * 128 # panels * n_pixels_x * n_pixels_y
    N_slices = 1000
    N_test_orientations = 10000
    ref_orientations = test_data['orientations'][:N_test_orientations]
    ref_orientations = np.reshape(ref_orientations, [N_test_orientations, 4])
    ac_support_size = 151
    oversampling = 1
    print(f'Test orientation matching N_slices: {N_slices} N_test_orientations:{N_test_orientations}')

    # h5py stores panels, det_x, det_y, dimension (4, 512, 512, 3)
    # need to move dimension to the first column (3, 4, 512, 512)
    pixel_position_reciprocal = np.moveaxis(
            test_data['pixel_position_reciprocal'][:], -1, 0)

    # Load everything into the data class
    inp = InputData(test_case, test_data['intensities'][:N_slices],
            pixel_position_reciprocal,
            test_data['pixel_index_map'],
            test_data['orientations'],
            test_data['volume'],
            ref_orientations,
            n_pixels, 
            ac_support_size,
            oversampling,
            )


    slices_ = inp.slices_
    pixel_position_reciprocal = inp.get_pixel_position_reciprocal()
    pixel_index_map = inp.pixel_index_map
    known_quaternions = inp.known_quaternions
    ac_phased = inp.ac_phased
    ref_orientations = inp.ref_orientations
    N_pixels = inp.n_pixels
    ac_support_size = inp.ac_support_size
    oversampling = inp.oversampling
    
    # calculate pixel distance to get max
    pixel_distance_reciprocal = prep.compute_pixel_distance(
        pixel_position_reciprocal)
    reciprocal_extent = pixel_distance_reciprocal.max()

    print(f'Input data:')
    print(f'  test case: {inp.name}')
    print(f'  slices_: {slices_.shape} dtype: {slices_.dtype}')
    print(f'  pixel_position_reciprocal: {pixel_position_reciprocal.shape} dtype: {pixel_position_reciprocal.dtype}')
    print(f'  pixel_distance_reciprocal: {pixel_distance_reciprocal.shape} dtype: {pixel_distance_reciprocal.dtype}')
    print(f'  pixel_index_map: {pixel_index_map.shape} dtype: {pixel_index_map.dtype}')
    print(f'  reciprocal_extent: {reciprocal_extent} dtype: {type(reciprocal_extent)}')
    print(f'  ac_phased: {ac_phased.shape} dtype: {ac_phased.dtype}')
    print(f'  ref_orientations: {ref_orientations.shape} dtype: {ref_orientations.dtype}')
    print(f'  N_pixels: {N_pixels}')
    print(f'  oversampling: {oversampling}')
    print(f'  ac_support_size: {ac_support_size}')


    # call orientation matching
    calc_quaternions = match(
        ac_phased, slices_,
        pixel_position_reciprocal, 
        pixel_distance_reciprocal,
        ref_orientations=ref_orientations)

    eps = 1e-2
    cn_pass = 0
    for i in range(N_slices):
        a = known_quaternions[i]
        b = calc_quaternions[i]
        print(a, b, abs(np.dot(a,b)))
        if abs(np.dot(a,b)) > 1-eps:
            cn_pass += 1
    success_rate = (cn_pass/N_slices)
    print(f'N_slices:{N_slices} Pass:{cn_pass} Success Rate:{success_rate*100:.2f}%')
    assert success_rate > 0.9
    
def run_match():
    test_match("3iyf") 
    test_match("2cex")


if __name__ == "__main__":
    run_match()
