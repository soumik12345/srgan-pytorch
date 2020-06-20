import math
import torch
from .blocks import ResidualBlock, UpSampleBlock, DiscriminatorBlock


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


class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            DiscriminatorBlock(
                in_channels=64, out_channels=64,
                kernel_size=3, stride=2, padding=1
            ),
            DiscriminatorBlock(
                in_channels=64, out_channels=128,
                kernel_size=3, stride=1, padding=1
            ),
            DiscriminatorBlock(
                in_channels=128, out_channels=128,
                kernel_size=3, stride=2, padding=1
            ),
            DiscriminatorBlock(
                in_channels=128, out_channels=256,
                kernel_size=3, stride=1, padding=1
            ),
            DiscriminatorBlock(
                in_channels=256, out_channels=256,
                kernel_size=3, stride=2, padding=1
            ),
            DiscriminatorBlock(
                in_channels=256, out_channels=512,
                kernel_size=3, stride=1, padding=1
            ),
            DiscriminatorBlock(
                in_channels=512, out_channels=512,
                kernel_size=3, stride=2, padding=1
            ),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=1024, kernel_size=1
            ),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(
                in_channels=1024,
                out_channels=1, kernel_size=1
            )
        )

    def forward(self, x):
        batch_size = x.size(0)
        y = self.net(x)
        y = y.view(batch_size)
        y = torch.sigmoid(y)
        return y
