import PyNVTX as nvtx

from spinifel.sequential.orientation_matching import (
    slicing_and_match as sequential_match,
)

from spinifel.sequential.orientation_matching import SNM

class SNM_MPI(SNM):
    def __init__(
        self,
        settings,
        slices_,
        pixel_position_reciprocal,
        pixel_distance_reciprocal,
        nufft,
    ):
        super().__init__(settings,
        slices_, pixel_position_reciprocal,
        pixel_distance_reciprocal, nufft)

        


@nvtx.annotate("mpi/orientation_matching.py", is_prefix=True)
def match(
    ac,
    slices_,
    pixel_position_reciprocal,
    pixel_distance_reciprocal,
    ref_orientations=None,
):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    return sequential_match(
        ac,
        slices_,
        pixel_position_reciprocal,
        pixel_distance_reciprocal,
        ref_orientations=ref_orientations,
    )
