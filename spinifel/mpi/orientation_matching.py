import numpy as np
import PyNVTX as nvtx

from spinifel import parms, contexts
from spinifel.sequential.orientation_matching import slicing_and_match as sequential_match


@nvtx.annotate("mpi/orientation_matching.py", is_prefix=True)
def match(ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal, ref_orientations):
    # The reference orientations don't have to match exactly between ranks.
    # Each rank aligns its own slices.
    # We can call the sequential function on each rank, provided that the
    # cost of generating the model_slices isn't prohibitive.
    comm = contexts.comm
    size = comm.Get_size()
    rank = comm.Get_rank()

    color = rank // 6
     
    newcomm = comm.Split(color)
    newcomm_rank = newcomm.rank
    newcomm_size = newcomm.size

    N_orientations = parms.N_orientations
    N_images_per_rank = parms.N_images_per_rank

    orientations_matched_local, minDist_local = sequential_match(
        ac, slices_, pixel_position_reciprocal, pixel_distance_reciprocal, ref_orientations)
    
    match_summary = newcomm.gather((newcomm_rank, orientations_matched_local, minDist_local), root=0)

    if newcomm_rank % 6 == 0:
        ranks, orientations_matched, minDist = [np.array(el) for el in zip(*match_summary)]
        print('ranks =', ranks)
        print('orientations_matched.shape =', orientations_matched.shape)
        print('minDist.shape =', minDist.shape)
        index = np.argmin(minDist, axis=0)
        orientations_matched = np.swapaxes(orientations_matched, 0, 1)
        for i in range(len(index)):
            orientations_selected = orientations_matched[i,index,:]   
    else:
        orientations_selected = None

    orientations_selected = newcomm.bcast(orientations_selected, root=0)     
    print('orientations_selected.shape =', orientations_selected.shape)    

    newcomm.Free()
    return orientations_selected
