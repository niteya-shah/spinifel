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
orientation_type_str = "float32"
volume_type_str = "complex64"
volume_shape = (151, 151, 151)
oversampling = 1

solve_ac_maxiter = 100

data_dir  = settings.data_dir

assert settings.data_filename, "Hdf5 filename input missing. Set DATA_FILENAME to the name of your hdf5 file."

data_path = data_dir / settings.data_filename

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
    N_images_per_rank = 100 * data_multiplier
    N_clipping = 0
    N_binning = 0
    N_orientations = 500
    N_batch_size = 100
    Mquat = int(oversampling * 20)  # 1/4 of uniform grid size
    N_generations = 5
    nER = 2
    nHIO = 3
    N_phase_loops = 5
else:
    N_images_per_rank = 1000 * data_multiplier
    N_clipping = 0
    N_binning = 0
    N_orientations = 3000 # model_slices
    N_batch_size = 100
    Mquat = int(oversampling * 20)  # 1/4 of uniform grid size
    N_generations = 10
    nER = 50
    nHIO = 25
    N_phase_loops = 10

M = 4 * Mquat + 1
M_ups = 2*M  # Upsampled grid for AC convolution technique
N_binning_tot = N_clipping + N_binning
reduced_det_shape = det_shape[:-2] + (
    det_shape[-2] // 2**N_binning_tot, det_shape[-1] // 2**N_binning_tot)
reduced_pixel_position_shape = (3,) + reduced_det_shape
reduced_pixel_index_shape = (2,) + reduced_det_shape
