import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.unet.unet import UNet
from training.diffusion import train_diffusion_model

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32),
        transforms.Normalize((0.5,), (0.5,))
    ])

    fashion_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_loader = DataLoader(fashion_mnist, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    unet = UNet(in_channels=1, out_channels=1, hidden_channels=[8, 16, 32, 64])

    train_diffusion_model(unet, train_loader)
