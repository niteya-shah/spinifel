import numpy     as np
import time

import spinifel.sequential.nearest_neighbor as nn


def match(slices_, model_slices, ref_orientations, batch_size=None):
    """ 
    Returns orientations which give the best match between slices_ and model_slices
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
