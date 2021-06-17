import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm, SymLogNorm

from spinifel import parms


def show_image(pixel_index_map, image, filename):
    # load image data
    buffer = np.zeros((pixel_index_map[0].max()+1, pixel_index_map[1].max()+1),
                      dtype=image.dtype)
    buffer[pixel_index_map[0], pixel_index_map[1]] = image
    # set all values that equal 0 to NaN => will render as white on the log plot
    buffer[buffer == 0] = np.nan

    # plot image data
    plt.imshow(buffer, norm=LogNorm())
    plt.colorbar()
#    plt.savefig(parms.out_dir / filename)
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
    norm = SymLogNorm(1e-3, base=10, vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(figsize=(8.0, 2.5), nrows=1, ncols=3)
    axes[0].imshow(ac_midx, norm=norm)
    axes[1].imshow(ac_midy, norm=norm)
    axes[2].imshow(ac_midz, norm=norm)

    fig.colorbar(cm.ScalarMappable(norm=norm),
                 ax=axes.ravel().tolist(), fraction=0.04, pad=0.04)

    plt.savefig(parms.out_dir / filename)
    plt.close('all')
