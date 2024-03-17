import torch
import torch.nn as nn

from model.unet.unet import sinusoidal_embedding, make_te
from model.residual_block.residual_block import ResidualBlock


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_steps=1000, time_embedding_dim=100):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res_block_1 = ResidualBlock(out_channels, out_channels)
        self.res_block_2 = ResidualBlock(out_channels, out_channels)
        self.time_embedding = make_te(time_embedding_dim, in_channels)
        self.time_embed = nn.Embedding(n_steps, time_embedding_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_embedding_dim)
        self.time_embed.requires_grad_(False)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t = self.time_embed(time)
        time = self.time_embedding(t).reshape(x.shape[0], -1, 1, 1)
        x = self.down_conv(x + time)
        residual = self.res_block_1(x)
        x = self.res_block_2(x)
        return x, residual
