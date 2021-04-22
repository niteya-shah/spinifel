# Test match() function for orientation matching module
# Note that all three parallelization types (sequential, mpi, or legion)
# call the same match() function.
# 
# To run the test, use one of the jsrun commands in scripts/run_summit_mult.sh
# For orientation matching test,
# CPU:          ./scripts/run_summit_mult.sh -m -n 1 -a 1 -g 1 -r 1 -d 1 -s
# GPU:          ./scripts/run_summit_mult.sh -m -n 1 -a 1 -g 1 -r 1 -d 1 -s -c
# GPU+cufinufft:./scripts/run_summit_mult.sh -m -n 1 -a 1 -g 1 -r 1 -d 1 -s -c -f


# POSTMORTEM 
# MODULE: slicing
# - I/O handling in cmtip for parallel processing
# - H,K,L are converted to single precision and there's no oversampling parameter in cmtip
# - Spinifel uses finufftpy (Elliott's fork) and cmtip uses original finufft.
#   These two packages have different interfaces.
# - Calculating H,K,L from Rotation Matrix and Pixel Position in Reciprocal space is 
#    different in cmtip and spinifel. 
#    pixel_position_reciprocal format
#    cmtip 3 x 1 x N_pixels
#    spinifel 3 x 4 (panels) x N_pixels_per_panel
# - Precision control/selection is missing. Mixed precisions exist in both cmtip and spinifel
#    Found RuntimeError:
#    RuntimeError: FINUFFT data type must be complex128 for double precision, data may have mixed precision types
# - Spinifel mpi input handling bcast shared data
# - Random Rotation matrix calculated from skopi is transposed in cmtip


from spinifel.sequential.orientation_matching import match 
from spinifel.slicing import gen_model_slices_batch
import skopi as skp
import h5py
import pickle
import numpy as np
import os
from spinifel import prep

N_slices = 3
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
    if test_case == "3iyf":
        # Input data from this test was generated from cmtip and saved to 
        # a pickle file. 
        # NOTE:
        # known_orientations is the correct orientations (3)
        # ref_orientations is 100 random orientations mixed in with the correct ones (3)
        with open(os.path.join(test_data_dir, '3IYF', 'cmtip_3iyf_3.pickle'),'rb') as f:
            test_data = pickle.load(f)
            inten_data = pickle.load(f)
            n_pixels = 1 * 128 * 128 # panels * n_pixels_x * n_pixels_y
            ac_support_size = 151
            oversampling = 1
            batch_size = 103
            inp = InputData(test_case, test_data['intensities'],
                    test_data['pixel_position_reciprocal'],
                    test_data['known_orientations'],
                    test_data['volume'],
                    inten_data['ref_orientations'],
                    n_pixels,
                    ac_support_size,
                    oversampling,
                    batch_size)

    elif test_case == "2cex":
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
        pixel_position_reciprocal = pixel_position_reciprocal.reshape(
                (pixel_position_reciprocal.shape[0], 1, n_pixels))
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
    model_slices = gen_model_slices_batch(ac_phased, ref_orientations, 
            pixel_position_reciprocal, reciprocal_extent, oversampling, 
            ac_support_size, N_pixels, batch_size=N_batch_size)
    model_slices = model_slices.reshape((ref_orientations.shape[0], N_pixels))
    
    print(f'Input data:')
    print(f'  test case: {inp.name}')
    print(f'  slices_: {slices_.shape} dtype: {slices_.dtype}')
    print(f'  pixel_position_reciprocal: {pixel_position_reciprocal.shape} dtype: {pixel_position_reciprocal.dtype}')
    print(f'  pixel_distance_reciprocal: {pixel_distance_reciprocal.shape} dtype: {pixel_distance_reciprocal.dtype}')
    print(f'  reciprocal_extent: {reciprocal_extent} dtype: {type(reciprocal_extent)}')
    
    print(f'  ac_phased: {ac_phased.shape} dtype: {ac_phased.dtype}')
    print(f'  model_slices: {model_slices.shape} dtype: {model_slices.dtype}')
    print(f'  ref_orientations: {ref_orientations.shape} dtype: {ref_orientations.dtype}')
    print(f'  N_pixels: {N_pixels}')
    print(f'  N_batch_size: {N_batch_size}')
    print(f'  oversampling: {oversampling}')
    print(f'  ac_support_size: {ac_support_size}')


    # scale model_slices
    data_model_scaling_ratio = slices_.std() / model_slices.std()
    print(f"Data/Model std ratio: {data_model_scaling_ratio}.", flush=True)
    model_slices *= data_model_scaling_ratio
    
    # call orientation matching
    calc_quaternions = match(slices_, model_slices, ref_orientations, batch_size=N_batch_size)

    for i in range(N_slices):
        a = known_quaternions[i]
        b = calc_quaternions[i]
        print(a, b, np.dot(a,b))
        assert np.dot(a,b) - 1. < 1e-12


if __name__ == "__main__":
    test_match("3iyf")
    test_match("2cex")
