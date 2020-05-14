import os
from pathlib import Path


det_shape = (4, 512, 512)
N_images = 10000
data_field_name = "intensities"
data_type_str = "float32"

data_dir = Path(os.environ.get("DATA_DIR", ""))
data_path = data_dir / "2CEX-10k-2.h5"
