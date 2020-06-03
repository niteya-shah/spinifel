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
N_clipping = 1
N_binning = 4

data_dir = Path(os.environ.get("DATA_DIR", ""))
data_path = data_dir / "2CEX-10k-2.h5"

out_dir = Path(os.environ.get("OUT_DIR", ""))

if os.environ.get("SMALL_PROBLEM") == "1":
    N_images = 10
else:
    N_images = 1000
