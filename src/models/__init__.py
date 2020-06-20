import math
import torch
from .blocks import ResidualBlock, UpSampleBlock


class Generator(torch.nn.Module):

    def __init__(self, scale):
        super(Generator, self).__init__()
        n_upsample_blocks = int(math.log(scale, 2))

        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=64,
                kernel_size=9, padding=4
            ),
            torch.nn.PReLU()
        )

        res_blocks = [ResidualBlock(channels=64)] * 5
        self.block_2 = torch.nn.Sequential(*res_blocks)

        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(num_features=64)
        )

        upsample_blocks = [UpSampleBlock(channels=64, scale=2)] * n_upsample_blocks
        upsample_blocks += [
            torch.nn.Conv2d(
                in_channels=64, out_channels=3,
                kernel_size=9, padding=4
            )
        ]
        self.block_4 = torch.nn.Sequential(*upsample_blocks)

    def forward(self, x):
        block_1 = self.block_1(x)
        block_2 = self.block_2(block_1)
        block_3 = self.block_3(block_2)
        block_4 = self.block_4(block_1 + block_3)
        y = torch.tanh(block_4)
        return (y + 1) / 2
