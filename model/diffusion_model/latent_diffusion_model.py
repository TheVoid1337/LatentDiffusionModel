import torch
import torch.nn as nn
from tqdm import tqdm

from model.unet.unet import UNet
from model.encoder.encoder import Encoder
from model.decoder.decoder import Decoder


class LatentDiffusionModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int,
                 encoder_dims: list[int], decoder_dims: list[int], unet_dims: list[int]):
        super(LatentDiffusionModel, self).__init__()
        self.encoder = Encoder(input_dim, encoder_dims, latent_dim)
        self.unet = UNet(latent_dim, unet_dims, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_dims, output_dim)

    def forward(self, x: torch.Tensor, scheduler) -> torch.Tensor:

        latents = self.encoder(x)

        noise = torch.randn(latents.shape, device=x.device)

        for t in tqdm(scheduler.timesteps, disable=True, leave=False):
            x_noisy = scheduler.add_noise(latents, noise, t)

            time = torch.randint(0, len(scheduler.timesteps), (x.shape[0],)).to(x.device)

            pred_noise = self.unet(x_noisy, time)

            latents = scheduler.step(pred_noise, t, x_noisy).prev_sample

        pred_images = self.decoder(latents)

        return pred_images
