import matplotlib
import numpy
import os
from pathlib import Path

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

data_dir = Path(os.environ.get("DATA_DIR", ""))
data_path = data_dir / "2CEX-10k-2.h5"

out_dir = Path(os.environ.get("OUT_DIR", ""))

data_multiplier = int(os.environ.get("DATA_MULTIPLIER", 1))

if os.environ.get("SMALL_PROBLEM") == "1":
    N_images_per_rank = 10 * data_multiplier
    nER = 10
    nHIO = 5
    N_phase_loops = 5
    N_clipping = 1
    N_binning = 4
    Mquat = int(oversampling * 10)  # 1/4 of uniform grid size
else:
    N_images_per_rank = 1000 * data_multiplier
    nER = 50
    nHIO = 25
    N_phase_loops = 10
    N_clipping = 0
    N_binning = 3
    Mquat = int(oversampling * 20)  # 1/4 of uniform grid size

N_binning_tot = N_clipping + N_binning
reduced_det_shape = det_shape[:-2] + (
    det_shape[-2] // 2**N_binning_tot, det_shape[-1] // 2**N_binning_tot)
reduced_pixel_position_shape = (3,) + reduced_det_shape
reduced_pixel_index_shape = (2,) + reduced_det_shape
