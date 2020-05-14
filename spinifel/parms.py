import os
from pathlib import Path


det_shape = (4, 512, 512)
N_images = 1000


data_dir = Path(os.environ.get("DATA_DIR", ""))
data_path = data_dir / "2CEX-1.h5"
