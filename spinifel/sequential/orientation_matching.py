import numpy     as np
import skopi     as skp
import time

from   spinifel import parms, utils, autocorrelation, SpinifelSettings
import spinifel.sequential.nearest_neighbor as nn


settings = SpinifelSettings()



def match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal):
    """ 
    TODO MONA: add batch_size to avoid running out of memory 
    when no. of reference orientations increase. 

    Note that N_pixels also grown when N_binning is small but 
    it's harder to divide an image up at the moment.

    For each batch in batched ref. orientations
        get H, K, L for this batch
        generate model_slices for this H, K, L
        get indices of the best matched 
        
    """
    st = time.time()
    Mquat = parms.Mquat
    M = 4 * Mquat + 1
    N_orientations = parms.N_orientations
    N_pixels = utils.prod(parms.reduced_det_shape)
    N_slices = slices_.shape[0]
    assert slices_.shape == (N_slices,) + parms.reduced_det_shape
    N = N_pixels * N_orientations

    if not N_slices:
        return np.zeros((0, 4))

    ref_orientations = skp.get_uniform_quat(N_orientations, True)
    ref_rotmat = np.array([skp.quaternion2rot3d(quat) for quat in ref_orientations])
    H, K, L = np.einsum("ijk,klmn->jilmn", ref_rotmat, pixel_position_reciprocal)
    real_extent = 2
    reciprocal_extent = pixel_distance_reciprocal.max()
    H_ = H.flatten() / reciprocal_extent * np.pi / parms.oversampling
    K_ = K.flatten() / reciprocal_extent * np.pi / parms.oversampling
    L_ = L.flatten() / reciprocal_extent * np.pi / parms.oversampling
    
    st_slice = time.time()
    
    model_slices = autocorrelation.forward(
        ac, H_, K_, L_, 1, M, N, reciprocal_extent, True).real
    
    en_slice = time.time()

    # Imaginary part ~ numerical error
    model_slices = model_slices.reshape((N_orientations, N_pixels))
    slices_ = slices_.reshape((N_slices, N_pixels))
    data_model_scaling_ratio = slices_.std() / model_slices.std()
    print(f"Data/Model std ratio: {data_model_scaling_ratio}.", flush=True)
    model_slices *= data_model_scaling_ratio
    
    st_match = time.time()
    index = nn.nearest_neighbor(model_slices, slices_)
    
    en_match = time.time()

    print(f"Match tot:{en_match-st:.2f}s. slice={en_slice-st_slice:.2f}s. match={en_match-st_match:.2f}s. slice_oh={st_slice-st:.2f}s. match_oh={st_match-en_slice:.2f}s.")
    return ref_orientations[index]
