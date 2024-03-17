from typing import List

import torch
import torch.nn as nn

from model.unet.encoder.down_conv import DownConv
from model.unet.unet import sinusoidal_embedding, make_te
from model.residual_block.residual_block import ResidualBlock


class Encoder(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: List[int], n_steps=1000, time_embedding_dim=100):
        super(Encoder, self).__init__()

        self.input_conv_1 = nn.Conv2d(input_channels, hidden_channels[0], kernel_size=3, stride=1, padding=1)
        self.input_conv_2 = ResidualBlock(hidden_channels[0], hidden_channels[0])
        self.input_conv_3 = ResidualBlock(hidden_channels[0], hidden_channels[0])
        self.time_embed = nn.Embedding(n_steps, time_embedding_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_embedding_dim)
        self.time_embed.requires_grad_(False)
        self.input_time_embedding = make_te(time_embedding_dim, input_channels)

        self.encoder = nn.ModuleList([
            DownConv(hidden_channels[i - 1], hidden_channels[i]) for i in range(1, len(hidden_channels))
        ])

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> (torch.Tensor, List[torch.Tensor]):
        res_cons = []

        t = self.time_embed(time)
        t = self.input_time_embedding(t).reshape(x.shape[0], -1, 1, 1)
        x = self.input_conv_1(x + t)
        x = self.input_conv_2(x)
        x = self.input_conv_3(x)

        res_cons.append(x)

        for i, encoder_block in enumerate(self.encoder):
            x, res_con = encoder_block(x, time)
            if i != len(self.encoder) - 1:
                res_cons.append(res_con)

        return x, res_cons
