import torch
import torch.nn as nn

from model.unet.embedding import make_te, sinusoidal_embedding
from model.residual_block.residual_block import ResidualBlock


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_steps=1000, time_embedding_dim=100):
        super(UpConv, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res_block_1 = ResidualBlock(in_channels, out_channels)
        self.res_block_2 = ResidualBlock(out_channels, out_channels)
        self.time_embedding = make_te(time_embedding_dim, in_channels)
        self.time_embed = nn.Embedding(n_steps, time_embedding_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_embedding_dim)
        self.time_embed.requires_grad_(False)

    def forward(self, x: torch.Tensor, res_con: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(time)
        x = self.up_conv(x)
        x = torch.cat([x, res_con], dim=1)
        x = x + self.time_embedding(t).reshape(x.shape[0], -1, 1, 1)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        return x
