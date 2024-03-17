import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def plot_images(images: torch.Tensor, title: str = ""):
    images = images.detach().cpu()

    images = make_grid(images, nrow=8).permute(0, 2, 3, 1).numpy()

    plt.figure(figsize=(10, 10))
    plt.title(title)
    if images.shape[-1] == 3:
        plt.imshow(images)
    else:
        plt.imshow(images, cmap="gray")

    plt.show()
    plt.close()
