from typing import List

import torch
import torch.nn as nn

from model.unet.decoder.decoder import Decoder
from model.unet.embedding import sinusoidal_embedding, make_te
from model.unet.encoder.encoder import Encoder
from model.residual_block.residual_block import ResidualBlock


class UNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int], out_channels: int, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        self.encoder = Encoder(in_channels, hidden_channels, n_steps, time_emb_dim)

        self.bottleneck_conv1 = ResidualBlock(hidden_channels[-1], hidden_channels[-1])
        self.bottleneck_conv2 = ResidualBlock(hidden_channels[-1], hidden_channels[-1])

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        self.mid_embedding = make_te(time_emb_dim, hidden_channels[-1])

        hidden_channels.reverse()
        self.decoder = Decoder(hidden_channels, out_channels, n_steps, time_emb_dim)

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(time_step)

        x, skip_cons = self.encoder(x, time_step)

        time = self.mid_embedding(t).reshape(x.shape[0], -1, 1, 1)
        x = x + time
        x = self.bottleneck_conv1(x)
        x = self.bottleneck_conv2(x)
        x = self.decoder(x, skip_cons, time_step)

        return x
