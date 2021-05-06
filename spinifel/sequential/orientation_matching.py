import numpy     as np
import time

import spinifel.sequential.nearest_neighbor as nn
import logging
from   spinifel import parms, utils, autocorrelation, SpinifelSettings
import skopi as skp


def match(slices_, model_slices, ref_orientations, batch_size=None):
    """ 
    Determine orientations of the data images (slices_) by minimizing the euclidean distance 
    with the reference images (model_slices) and return orientations which give the best match.
   
    :param slice_: data images
    :param mode_slices: reference images
    :param ref_orientations: referene orientations
    :param batch_size: batch size
    :return ref_orientations: array of quaternions matched to slices_
    """
    
    if batch_size is None:
        batch_size = model_slices.shape[0]
    
    N_slices = slices_.shape[0]
    # TODO move this up to main level
    #assert slices_.shape == (N_slices,) + parms.reduced_det_shape

    if not N_slices:
        return np.zeros((0, 4))

    index = nn.nearest_neighbor(model_slices, slices_, batch_size)

    return ref_orientations[index]

def slicing_and_match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal):
    """
    Determine orientations of the data images by minimizing the euclidean distance with the reference images 
    computed by randomly slicing through the autocorrelation.
    MONA: This is a current hack to support Legion. For MPI, slicing is done separately 
    from orientation matching.

    :param ac: autocorrelation of the current electron density estimate
    :param slices_: data images
    :param pixel_position_reciprocal: pixel positions in reciprocal space
    :param pixel_distance_reciprocal: pixel distance in reciprocal space
    :return ref_orientations: array of quaternions matched to slices_
    """
    st_init = time.monotonic()
    logger = logging.getLogger(__name__)
    Mquat = parms.Mquat
    M = 4 * Mquat + 1
    N_orientations = parms.N_orientations
    N_batch_size = parms.N_batch_size
    N_pixels = utils.prod(parms.reduced_det_shape)
    N_slices = slices_.shape[0]
    assert slices_.shape == (N_slices,) + parms.reduced_det_shape
    N = N_pixels * N_orientations

    if not N_slices:
        return np.zeros((0, 4))

    ref_orientations = skp.get_uniform_quat(N_orientations, True)
    ref_rotmat = np.array([skp.quaternion2rot3d(quat) for quat in ref_orientations])
    reciprocal_extent = pixel_distance_reciprocal.max()

    # Calulate Model Slices in batch
    assert N_orientations % N_batch_size == 0, "N_orientations must be divisible by N_batch_size"
    slices_ = slices_.reshape((N_slices, N_pixels))
    model_slices_new = np.zeros((N,))
    
    st_slice = time.monotonic()

    for i in range(N_orientations//N_batch_size):
        st = i * N_batch_size
        en = st + N_batch_size
        H, K, L = np.einsum("ijk,klmn->jilmn", ref_rotmat[st:en], pixel_position_reciprocal)
        H_ = H.flatten() / reciprocal_extent * np.pi / parms.oversampling
        K_ = K.flatten() / reciprocal_extent * np.pi / parms.oversampling
        L_ = L.flatten() / reciprocal_extent * np.pi / parms.oversampling
        N_batch = N_pixels * N_batch_size
        st_m = i * N_batch_size * N_pixels
        en_m = st_m + (N_batch_size * N_pixels)
        model_slices_new[st_m:en_m] = autocorrelation.forward(
                ac, H_, K_, L_, 1, M, N_batch, reciprocal_extent, True).real
        
    en_slice = time.monotonic()
    
    # Imaginary part ~ numerical error
    model_slices_new = model_slices_new.reshape((N_orientations, N_pixels))
    data_model_scaling_ratio = slices_.std() / model_slices_new.std()
    print(f"New Data/Model std ratio: {data_model_scaling_ratio}.", flush=True)
    model_slices_new *= data_model_scaling_ratio
    
    # Calculate Euclidean distance in batch to avoid running out of GPU Memory
    st_match = time.monotonic()
    index = nn.nearest_neighbor(model_slices_new, slices_, N_batch_size)
    en_match = time.monotonic()

    print(f"Match tot:{en_match-st_init:.2f}s. slice={en_slice-st_slice:.2f}s. match={en_match-st_match:.2f}s. slice_oh={st_slice-st_init:.2f}s. match_oh={st_match-en_slice:.2f}s.")
    return ref_orientations[index]
