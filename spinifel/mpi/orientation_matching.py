import PyNVTX as nvtx

from spinifel.sequential.orientation_matching import slicing_and_match as sequential_match



@nvtx.annotate("mpi/orientation_matching.py", is_prefix=True)
def match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal, rank):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    return sequential_match(
        ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal, rank)
