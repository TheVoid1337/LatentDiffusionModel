import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.diffusion_model.latent_diffusion_model import LatentDiffusionModel
from training import train_model
from diffusers.schedulers import DDPMScheduler

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5,), (0.5,))
    ])

    fashion_mnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    train_loader = DataLoader(fashion_mnist, batch_size=1, shuffle=True, num_workers=2)

    model = LatentDiffusionModel(1, 1, 100, [16],
                                 [16], [16, 32, 64, 128])

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    train_model(scheduler, model, 10, train_loader, "weights/diffusion_model.pth")
