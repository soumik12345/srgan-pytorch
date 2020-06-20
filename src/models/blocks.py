import torch


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=3, padding=1
        )
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=channels)
        self.activation = torch.nn.PReLU()
        self.conv_2 = torch.nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=3, padding=1
        )
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        y = self.conv_1(x)
        y = self.batch_norm_1(y)
        y = self.activation(y)
        y = self.conv_2(y)
        y = self.batch_norm_2(y)
        return x + y


class UpSampleBlock(torch.nn.Module):

    def __init__(self, channels, scale):
        super(UpSampleBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels * (scale ** 2),
            kernel_size=3, padding=1
        )
        self.pixel_shuffle = torch.nn.PixelShuffle(scale)
        self.activation = torch.nn.PReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.pixel_shuffle(y)
        y = self.activation(y)
        return y
