import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


# def plot_images(images: torch.Tensor, title: str = ""):
#     images = images.detach().cpu()
#
#     images = (make_grid(images, nrow=8).permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
#
#     plt.figure(figsize=(10, 10))
#     plt.title(title)
#     plt.imshow(images, cmap="gray")
#
#     plt.show()
#     plt.close()


def plot_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()