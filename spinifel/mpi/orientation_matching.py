from spinifel.sequential.orientation_matching import match as sequential_match


def match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    return sequential_match(
        ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal)
