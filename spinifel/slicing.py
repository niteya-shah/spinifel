import numpy   as np
import skopi   as skp
import PyNVTX  as nvtx

from spinifel import autocorrelation



@nvtx.annotate("slicing.py", is_prefix=True)
def gen_model_slices(ac, ref_orientations,
                     pixel_position_reciprocal, reciprocal_extent,
                     oversampling, ac_support_size, N_pixels,
                     override_forward_with):
    """
    Generate model slices using given reference orientations (in quaternion)
    """

    N_orientations = ref_orientations.shape[0]

    # Get q points (normalized by recirocal extent and oversampling)
    H_, K_, L_ = autocorrelation.gen_nonuniform_normalized_positions(
        ref_orientations, pixel_position_reciprocal, reciprocal_extent, oversampling)

    # TODO: Add back arbitrary support size
    if ac_support_size is None:
        ac_support_size = ac.shape[0]

    N = N_pixels * N_orientations

    if override_forward_with is None:
        # This will use finufft or cufinufft depending on -f setting
        nuvect = autocorrelation.forward(
                 ac, H_, K_, L_, 1, ac_support_size, N, reciprocal_extent, True)
    elif override_forward_with == 'cpu':
        print(f'gen_model_slices override using forward_cpu')
        nuvect = autocorrelation.forward_cpu(
                 ac, H_, K_, L_, 1, ac_support_size, N, reciprocal_extent, True)
    elif override_forward_with == 'gpu':
        print(f'gen_model_slices override using forward_gpu')
        nuvect = autocorrelation.forward_gpu(
                 ac, H_, K_, L_, 1, ac_support_size, N, reciprocal_extent, True)

    model_slices = nuvect.real

    return model_slices



@nvtx.annotate("slicing.py", is_prefix=True)
def gen_model_slices_batch(
        ac,
        ref_orientations,
        pixel_position_reciprocal,
        reciprocal_extent,
        oversampling,
        ac_support_size,
        N_pixels,
        batch_size=None,
        override_forward_with=None):
    """
    Use batch_size parameter to create model_slices in batch
    This prevent out of memory when forward_gpu is used.
    """

    if batch_size is None:
        N_batch_size = ref_orientations.shape[0]
    else:
        N_batch_size = batch_size

    N_orientations = ref_orientations.shape[0]
    assert N_orientations % N_batch_size == 0, f"N_orientations ({N_orientations}) must be divisible by N_batch_size ({N_batch_size})"
    N = N_pixels * N_orientations

    model_slices_batch = np.zeros((N,))

    for i in range(N_orientations // N_batch_size):
        # calculate start - end indices for ref orientations
        st = i * N_batch_size
        en = st + N_batch_size

        # calculate start - end indices for model slices
        # because the returned values are 1D of model slices
        st_m = i * N_batch_size * N_pixels
        en_m = st_m + (N_batch_size * N_pixels)
        model_slices_batch[st_m:en_m] = gen_model_slices(ac,
                                                         ref_orientations[st:en],
                                                         pixel_position_reciprocal,
                                                         reciprocal_extent,
                                                         oversampling,
                                                         ac_support_size,
                                                         N_pixels,
                                                         override_forward_with)


    return model_slices_batch
