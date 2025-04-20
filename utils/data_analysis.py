import matplotlib.pyplot as plt
import numpy as np


def PlotT1T2Map(t1_gt, t2_gt, t1_hat, t2_hat, save_dir):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    im = axs[0, 0].imshow(t1_gt, vmin=0, vmax=4000,
                   cmap='jet')
    fig.colorbar(im, ax=axs[0, 0], orientation='vertical')
    im = axs[1, 0].imshow(t2_gt, vmin=0, vmax=250,
                   cmap='jet')
    fig.colorbar(im, ax=axs[1, 0], orientation='vertical')
    im = axs[0, 1].imshow(np.abs(t1_hat), vmin=0, vmax=4000,
                   cmap='jet')
    fig.colorbar(im, ax=axs[0, 1], orientation='vertical')
    im = axs[1, 1].imshow(np.abs(t2_hat), vmin=0, vmax=250,
                   cmap='jet')
    fig.colorbar(im, ax=axs[1, 1], orientation='vertical')
    fig.savefig(save_dir, bbox_inches='tight')
    plt.close(fig)