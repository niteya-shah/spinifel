import matplotlib
import numpy
import os
from pathlib import Path

from spinifel import SpinifelSettings

settings = SpinifelSettings()

matplotlib.use("Agg")
numpy.seterr(divide='ignore', invalid='ignore')

det_shape = (4, 512, 512)
N_images_max = 10000
data_field_name = "intensities"
data_type_str = "float32"
pixel_position_shape = (3,) + det_shape
pixel_position_type_str = "float32"
pixel_index_shape = (2,) + det_shape
pixel_index_type_str = "int32"
oversampling = 1

data_dir  = settings.data_dir
data_path = data_dir / "2CEX-10k-2.h5"
if settings.use_psana:
    use_psana = True
    exp = 'xpptut15'
    runnum = 1
else:
    use_psana = False

out_dir = settings.out_dir

data_multiplier = settings.data_multiplier
verbosity = settings.verbosity

if settings.small_problem:
    N_images_per_rank = 10 * data_multiplier
    nER = 10
    nHIO = 5
    N_phase_loops = 5
    N_clipping = 1
    N_binning = 4
    N_orientations = 1000
    Mquat = int(oversampling * 10)  # 1/4 of uniform grid size
else:
    N_images_per_rank = 1000 * data_multiplier
    nER = 50
    nHIO = 25
    N_phase_loops = 10
    N_clipping = 0
    N_binning = 3
    N_orientations = 2000
    Mquat = int(oversampling * 20)  # 1/4 of uniform grid size

M = 4 * Mquat + 1
M_ups = 2*M  # Upsampled grid for AC convolution technique
N_binning_tot = N_clipping + N_binning
reduced_det_shape = det_shape[:-2] + (
    det_shape[-2] // 2**N_binning_tot, det_shape[-1] // 2**N_binning_tot)
reduced_pixel_position_shape = (3,) + reduced_det_shape
reduced_pixel_index_shape = (2,) + reduced_det_shape
