from typing import List

import torch
from torch import nn

from model.unet.decoder.up_conv import UpConv


class Decoder(nn.Module):
    def __init__(self, hidden_channels: List[int], out_channels: int, n_steps=1000, time_embedding_dim=100):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList([
            UpConv(hidden_channels[i], hidden_channels[i + 1], n_steps, time_embedding_dim)
            for i in range(len(hidden_channels) - 1)
        ])

        self.output_conv = nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, res_cons: List[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        for block in self.decoder:
            x = block(x, res_cons.pop(), time)

        x = self.output_conv(x)
        return x
