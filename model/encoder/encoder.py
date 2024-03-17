from typing import List

import torch
from torch import nn

from model.encoder.downsample import DownConvBlock
from model.residual_block.residual_block import ResidualBlock


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_channels: List[int], latent_dim: int):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList([])
        self.input_conv = nn.Conv2d(input_size, hidden_channels[0], kernel_size=3, stride=2, padding=1)
        self.input_res_1 = ResidualBlock(hidden_channels[0], hidden_channels[0])
        self.input_res_2 = ResidualBlock(hidden_channels[0], hidden_channels[0])

        for i in range(1, len(hidden_channels)):
            self.encoder.append(DownConvBlock(hidden_channels[i - 1], hidden_channels[i]))

        self.last_conv_1 = nn.Conv2d(hidden_channels[-1], latent_dim, kernel_size=3, padding=1)
        self.last_conv_2 = nn.Conv2d(hidden_channels[-1], latent_dim, kernel_size=3, padding=1)

    def __reparameterize(self, mu, log_var):
        noise = torch.randn_like(mu)

        z = mu + torch.exp(0.5 * log_var) * noise

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.input_res_1(x)
        x = self.input_res_2(x)
        for block in self.encoder:
            x = block(x)

        mu = self.last_conv_1(x)
        log_var = self.last_conv_2(x)

        x = self.__reparameterize(mu, log_var)

        return x
