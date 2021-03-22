import numpy     as np
import skopi     as skp
import time
import logging

from   spinifel import parms, utils, autocorrelation, SpinifelSettings
import spinifel.sequential.nearest_neighbor as nn


settings = SpinifelSettings()


def match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal):
    """ 
    MONA: add batch_size to avoid running out of memory 
    when no. of reference orientations increase. 

    Note that N_pixels also growis when N_binning is small but 
    it's harder to divide an image up at the moment...

    Some pseudo:
    For each batch in batched ref. orientations
        get H, K, L for this batch
        generate model_slices for this H, K, L
        calculate euclidean distant for this batch
    Get indices of the orientations for all batches
        
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
    
    st_match = time.monotonic()

    # Calculate Euclidean distance in batch to avoid running out of GPU Memory
    euDistNew = np.zeros((slices_.shape[0], model_slices_new.shape[0]), dtype=np.float32)
    for i in range(model_slices_new.shape[0]//N_batch_size):
        st = i * N_batch_size
        en = st + N_batch_size
        euDistNew[:, st:en] = nn.calc_eudist(model_slices_new[st:en], slices_).reshape(slices_.shape[0], N_batch_size)
    euDistNew = euDistNew.flatten()

    indexNew = nn.calc_argmin(euDistNew, 
                        slices_.shape[0],
                        model_slices_new.shape[0],
                        slices_.shape[1])
    
    en_match = time.monotonic()

    print(f"Match tot:{en_match-st_init:.2f}s. slice={en_slice-st_slice:.2f}s. match={en_match-st_match:.2f}s. slice_oh={st_slice-st_init:.2f}s. match_oh={st_match-en_slice:.2f}s.")
    return ref_orientations[indexNew]
