import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.diffusion_model.latent_diffusion_model import LatentDiffusionModel
from torch.optim import Adam

from plotter.plotter import plot_images
from sampling.sampler import Sampler


def train_model(scheduler, model: LatentDiffusionModel, num_epochs: int, train_loader: DataLoader, file_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sampler = Sampler(scheduler)
    optim = Adam(model.parameters(), lr=10e-5)
    criterion = nn.MSELoss()
    model.train()

    for epoch in tqdm(range(num_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            optim.zero_grad()

            pred_images = model(images, scheduler)

            loss = criterion(pred_images, images)

            loss.backward()

            optim.step()

            epoch_loss += loss.item()

        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.6f}")
        images = sampler.sample_new_images(16, model, device)
        plot_images(images, "samples at epoch {}".format(epoch + 1))

    model_state = model.state_dict()

    torch.save(model_state, file_name)
