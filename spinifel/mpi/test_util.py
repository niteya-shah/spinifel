####################### Utilities for main and unit tests #######################
import os
import h5py
from spinifel import settings
import numpy as np


def get_known_orientations():
    """Returns known orientations for 3iyf
    
    Motivation: our main tests rely on convergence check of the cc between known
    electron density and the calculated one from spinifel. I observed that this
    can fail or pass depending on the starting point. By giving the module a better
    starting point (some known orientations), we maybe able to get reliable convergence
    that should be more CI friendly.
    """
    test_data_dir = os.environ.get("test_data_dir", "")
    N_test_orientations = settings.N_orientations

    # Open data file with correct answers

    test_data = h5py.File(os.path.join(test_data_dir, "3IYF", "3iyf_sim_10k.h5"), "r")

    # Get known orientations
    ref_orientations = test_data["orientations"][:N_test_orientations]
    ref_orientations = np.reshape(ref_orientations, [N_test_orientations, 4])

    return ref_orientations
