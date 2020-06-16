import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm

from spinifel import parms


def show_image(pixel_index_map, image, filename):
    buffer = np.zeros((pixel_index_map[0].max()+1, pixel_index_map[1].max()+1),
                      dtype=image.dtype)
    buffer[pixel_index_map[0], pixel_index_map[1]] = image
    plt.imshow(buffer, norm=LogNorm())
    plt.colorbar()
    plt.savefig(parms.out_dir / filename)
    plt.cla()
    plt.clf()


def show_volume(ac, Mquat, filename):
    ac_midz = ac[..., 2*Mquat]
    plt.imshow(ac_midz, norm=SymLogNorm(1e-3))
    plt.colorbar()
    plt.savefig(parms.out_dir / filename)
    plt.cla()
    plt.clf()
