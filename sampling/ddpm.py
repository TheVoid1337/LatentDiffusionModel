import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDPMScheduler


class DDPM:
    def __init__(self, device, num_steps=1000, start: float = 10e-4, end: float = 0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(start, end, num_steps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prod)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1.0 - self.alphas_cum_prod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cum_prod_prev = F.pad(self.alphas_cum_prod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alphas_cum_prod_prev) / (1. - self.alphas_cum_prod)

    def add_noise(self, x_0, time, noise=None):
        if noise is None:
            noise = torch.randn(x_0.shape, device=x_0.device)

        sqrt_alpha_cum_prod_t = self.sqrt_alphas_cum_prod[time]
        sqrt_one_minus_alpha_cum_prod_t = self.sqrt_one_minus_alphas_cum_prod[time]

        while len(sqrt_alpha_cum_prod_t.shape) < len(x_0.shape):
            sqrt_alpha_cum_prod_t = sqrt_alpha_cum_prod_t.unsqueeze(-1)

        while len(sqrt_one_minus_alpha_cum_prod_t.shape) < len(x_0.shape):
            sqrt_one_minus_alpha_cum_prod_t = sqrt_one_minus_alpha_cum_prod_t.unsqueeze(-1)

        return sqrt_alpha_cum_prod_t * x_0 + sqrt_one_minus_alpha_cum_prod_t * noise

    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]

        out = vals.gather(-1, t.squeeze())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def __sample_timestep(self, x, t, unet):
        time_tensor = (torch.ones(x.shape[0], 1, device=x.device, dtype=torch.int64) * t)

        betas_t = self.get_index_from_list(self.betas, time_tensor, x.shape)
        sqrt_one_minus_alphas_cum_prod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cum_prod, time_tensor, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, time_tensor, x.shape)

        posterior_variance_t = self.get_index_from_list(self.posterior_variance, time_tensor, x.shape)

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * unet(x, time_tensor) / sqrt_one_minus_alphas_cum_prod_t
        )

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, num_images, unet, device="cuda"):
        img = torch.randn((num_images, 1, 32, 32), device=device)

        for idx, t in enumerate(list(range(self.num_steps))[::-1]):
            t = torch.full((1,), t, device=device, dtype=torch.long)
            img = self.__sample_timestep(img, t, unet)

        return img
