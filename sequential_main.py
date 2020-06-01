import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from spinifel import parms


def get_saxs(pixel_position_reciprocal, slices):
    qs = np.sqrt((pixel_position_reciprocal**2).sum(axis=0)).flatten()
    qs /= qs.max()
    N = 100
    idx = (qs*N).astype(np.int)
    saxs_acc = np.bincount(idx, slices[:].sum(axis=0).flatten(), N)
    saxs_wgt = np.bincount(idx, None, N)
    saxs = saxs_acc / saxs_wgt
    return saxs


def show_image(pixel_index_map, image, filename):
    buffer = np.zeros((pixel_index_map[0].max()+1, pixel_index_map[1].max()+1),
                      dtype=image.dtype)
    buffer[pixel_index_map[0], pixel_index_map[1]] = image
    plt.imshow(buffer, norm=LogNorm())
    plt.savefig(parms.out_dir / filename)


def main():
    print("In sequential main", flush=True)

    N_images = 1
    det_shape = parms.det_shape

    with h5py.File(parms.data_path, 'r') as h5f:
        pixel_position_reciprocal = h5f['pixel_position_reciprocal'][:]
        pixel_index_map = h5f['pixel_index_map'][:]
        slices_ = h5f['intensities'][:N_images]
    pixel_position_reciprocal = np.moveaxis(pixel_position_reciprocal, -1, 0)
    pixel_index_map = np.moveaxis(pixel_index_map, -1, 0)

    show_image(pixel_index_map, slices_[0], "test.png")


if __name__ == '__main__':
    main()
