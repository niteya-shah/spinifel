# Test match() function for orientation matching module
# Note that all three parallelization types (sequential, mpi, or legion)
# call the same match() function.

from spinifel.sequential.orientation_matching import match 
import h5py
import pickle
import numpy as np

N_slices = 10

def test_match():
    
    # load known quaternions and test images
    f_data = h5py.File('/gpfs/alpine/proj-shared/chm137/data/spi/2CEX-10k-2.h5','r')
    known_quaternions = f_data['orientations'][:N_slices,:]


    # load pre-computed data
    with open('/gpfs/alpine/scratch/monarin/chm137/mona/2CEX-10k-2-for-test.h5', 'rb') as f_pre:
        test_data = pickle.load(f_pre)
        ac_phased = test_data['ac_phased']
        pixel_position_reciprocal = test_data['pixel_position_reciprocal']
        pixel_distance_reciprocal = test_data['pixel_distance_reciprocal']
        slices_ = test_data['slices_']

    # call orientation matching
    calc_quaternions = match(
        ac_phased, slices_,
        pixel_position_reciprocal, pixel_distance_reciprocal)


    for i in range(N_slices):
        a = known_quaternions[i]
        b = calc_quaternions[i]
        print(a, b, np.dot(a,b))



if __name__ == "__main__":
    test_match()
