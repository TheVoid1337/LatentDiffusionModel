import torch
from tqdm import tqdm

from model.diffusion_model.latent_diffusion_model import LatentDiffusionModel


class Sampler:
    def __init__(self, sampler, num_inference_steps: int = 50):
        self.sampler = sampler
        self.num_inference_steps = num_inference_steps

    def sample_new_images(self, num_images: int,
                          model: LatentDiffusionModel,
                          device: torch.device,
                          num_channels: int = 1,
                          latent_size: int = 28
                          ):
        self.sampler.set_timesteps(self.num_inference_steps)

        model.eval()

        with torch.no_grad():
            latents = torch.randn((num_images, num_channels, latent_size, latent_size)).to(device)
            for i in tqdm(reversed(range(0, self.sampler.timesteps)), desc='sampling loop time step',
                          total=self.sampler.timesteps, position=0):
                noise = torch.randn(latents.shape)
                time = torch.full((num_images,), i, device=device, dtype=torch.long)
                pred_noise = model.unet(latents, time)
                latents = self.sampler.step(pred_noise, i, noise).prev_sample

            return model.decoder(latents)




