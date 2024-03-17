import torch
from torch import nn

from model.residual_block.residual_block import ResidualBlock


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.input_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res_block_1 = ResidualBlock(out_channels, out_channels)
        self.res_block_2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        return x
