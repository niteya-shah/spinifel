import matplotlib
import numpy
import os
from pathlib import Path

from spinifel import SpinifelSettings

settings = SpinifelSettings()

matplotlib.use("Agg")
numpy.seterr(divide='ignore', invalid='ignore')

det_shape = settings.det_shape  # (1, 128, 128)
N_images_max = settings.N_images_max  # 10000
N_generations = settings.N_generations  # 10
data_field_name = settings.data_field_name  # "intensities"
data_type_str = settings.data_type_str  # "float32"
pixel_position_shape = settings.pixel_position_shape  # (3,) + det_shape
pixel_position_type_str = settings.pixel_position_type_str  # "float32"
pixel_index_shape = settings.pixel_index_shape  # (2,) + det_shape
pixel_index_type_str = settings.pixel_index_type_str  # "int32"
orientation_type_str = settings.orientation_type_str  # "float32"
volume_type_str = settings.volume_type_str  # "complex65"
volume_shape = settings.volume_shape  # (149, 149, 149)
oversampling = settings.oversampling  # 1

solve_ac_maxiter = settings.solve_ac_maxiter  # 100

data_dir  = settings.data_dir

assert settings.data_filename, "Hdf5 filename input missing. Set DATA_FILENAME to the name of your hdf5 file."

data_path = settings.data_path  # data_dir / settings.data_filename

chk_convergence = settings.chk_convergence

if settings.use_psana:
    use_psana = True
    exp = settings.ps_exp  # 'xpptut15'
    runnum = settins.ps_runnum  # 1
else:
    use_psana = False

out_dir = settings.out_dir

verbosity = settings.verbosity

N_images_per_rank = settings.n_images_per_rank
# if settings.small_problem:
#     nER = 10
#     nHIO = 5
#     N_phase_loops = 5
#     N_clipping = 1
#     N_binning = 4
#     N_orientations = 1000
#     N_batch_size = 1000
#     Mquat = int(oversampling * 20)  # 1/4 of uniform grid size
# else:
#     nER = 50
#     nHIO = 25
#     N_phase_loops = 10
#     N_clipping = 0
#     N_binning = 0
#     N_orientations = 200000 # model_slices
#     N_batch_size = 100
#     Mquat = int(oversampling * 20)  # 1/4 of uniform grid size
nER = settings.nER  # 10
nHIO = settings.nHIO  # 5
N_phase_loops = settings.N_phase_loops  # 5
N_clipping = settings.N_clipping  # 1
N_binning = settings.N_binning  # 4
N_orientations = settings.N_orientations  #  1000
N_batch_size = settings.N_batch_size  # 1000
Mquat = settings.Mquat  # int(oversampling * 20)  # 1/4 of uniform grid size

M = settings.M  # 4*Mquat + 1
M_ups = settings.M_ups  # 2*M  # Upsampled grid for AC convolution technique
N_binning_tot = settings.N_binning_tot  # N_clipping + N_binning
reduced_det_shape = settings.reduced_det_shape  # det_shape[:-2] + (
#     det_shape[-2] // 2**N_binning_tot, det_shape[-1] // 2**N_binning_tot)
reduced_pixel_position_shape = settings.reduced_pixel_position_shape  # (3,) + reduced_det_shape
reduced_pixel_index_shape = settings.reduced_pixel_index_shape  # (2,) + reduced_det_shape

# PSANA2
ps_smd_n_events = settings.ps_smd_n_events
ps_eb_nodes     = settings.ps_eb_nodes
ps_srv_nodes    = settings.ps_srv_nodes
