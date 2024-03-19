import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sampling.ddpm import DDPM
from plotter.plotter import plot_images


def train_diffusion_model(model, train_loader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model.load_state_dict(torch.load("weights/diffusion_model.pth"))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10e-5)

    ddpm = DDPM(device)

    for epoch in range(epochs):
        train_loss = 0.0

        for images, _ in tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{epochs}", position=0):
            images = images.to(device)

            eta = torch.randn(images.shape, device=device)

            time = torch.randint(0, ddpm.num_steps, (images.shape[0],), device=device, dtype=torch.int)

            x_noisy = ddpm.add_noise(images, time, eta)

            eta_theta = model(x_noisy, time)
            optimizer.zero_grad()
            loss = F.mse_loss(eta_theta, eta)

            loss.backward()

            optimizer.step()

            train_loss += loss

        print(f"Epoch {epoch + 1}/{epochs} | Loss {train_loss / len(train_loader):.6f}")
        images = ddpm.sample(64, model)
        plot_images(images, f"Images at epoch {epoch + 1}")

    torch.save(model.state_dict(), f"weights/diffusion_model.pth")
