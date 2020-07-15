import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
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
    if ac.dtype == np.bool_:
        ac = ac.astype(np.float)
    ac_midx = ac[2*Mquat, :, :]
    ac_midy = ac[:, 2*Mquat, :]
    ac_midz = ac[:, :, 2*Mquat]
    vmin = 0
    vmax = ac.max()

    fig, axes = plt.subplots(figsize=(8.0, 2.5), nrows=1, ncols=3)
    axes[0].imshow(ac_midx, norm=SymLogNorm(1e-3, base=10), vmin=vmin, vmax=vmax)
    axes[1].imshow(ac_midy, norm=SymLogNorm(1e-3, base=10), vmin=vmin, vmax=vmax)
    axes[2].imshow(ac_midz, norm=SymLogNorm(1e-3, base=10), vmin=vmin, vmax=vmax)

    fig.colorbar(cm.ScalarMappable(norm=SymLogNorm(1e-3, base=10, vmin=vmin, vmax=vmax)),
                 ax=axes.ravel().tolist(), fraction=0.04, pad=0.04)

    plt.savefig(parms.out_dir / filename)
    plt.close('all')
