import PyNVTX as nvtx

from spinifel.sequential.orientation_matching import match as sequential_match



@nvtx.annotate("mpi/orientation_matching.py", is_prefix=True)
def match(slices_, model_slices, ref_orientations, batch_size=None):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    return sequential_match(
        slices_, model_slices, ref_orientations, batch_size=batch_size) 
