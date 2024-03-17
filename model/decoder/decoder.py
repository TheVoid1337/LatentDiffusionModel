from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.decoder.upsample import UpConvBlock


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super(Decoder, self).__init__()
        self.input_conv_1 = nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1)
        self.input_conv_2 = nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=3, stride=1, padding=1)
        self.input_conv_3 = UpConvBlock(hidden_dims[0], hidden_dims[0])

        self.decoder = nn.ModuleList([])
        for i in range(1, len(hidden_dims)):
            self.decoder.append(UpConvBlock(hidden_dims[i - 1], hidden_dims[i]))

        self.output_conv = nn.Conv2d(hidden_dims[-1], output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv_1(x)
        x = self.input_conv_2(x)
        x = self.input_conv_3(x)
        for decoder in self.decoder:
            x = decoder(x)

        x = self.output_conv(x)

        return x
