import torch
import torch.nn as nn

from model.residual_block.residual_block import ResidualBlock


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.SiLU()
        self.res_block_1 = ResidualBlock(out_channels, out_channels)
        self.res_block_2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        return x
