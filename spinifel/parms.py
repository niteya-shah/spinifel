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
N_clipping = 1
N_binning = 4

Mquat = 10  # 1/4 of uniform grid size

data_dir = Path(os.environ.get("DATA_DIR", ""))
data_path = data_dir / "2CEX-10k-2.h5"

out_dir = Path(os.environ.get("OUT_DIR", ""))

if os.environ.get("SMALL_PROBLEM") == "1":
    N_images_per_rank = 10
    nER = 10
    nHIO = 5
    N_phase_loops = 5
else:
    N_images_per_rank = 1000
    nER = 100
    nHIO = 50
    N_phase_loops = 20

N_binning_tot = N_clipping + N_binning
reduced_det_shape = det_shape[:-2] + (
    det_shape[-2] // 2**N_binning_tot, det_shape[-1] // 2**N_binning_tot)
reduced_pixel_position_shape = (3,) + reduced_det_shape
reduced_pixel_index_shape = (2,) + reduced_det_shape
