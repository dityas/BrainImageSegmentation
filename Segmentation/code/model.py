import torch.nn as N


class DownConv(N.Module):

    """
        Input(Batch, Channel, Depth, Height, Width) => Conv() => Relu * 2 => MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_in = N.Conv3d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1,
                                stride=1)

        self.conv_out = N.Conv3d(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 padding=1,
                                 stride=1)

        self.relu = N.ReLU()
        self.max_pool = N.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_out(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class UNet(N.Module):

    """
        Defines U-Net using conv and upconv blocks defined above.
    """

    def __init__(self):
        super().__init__()
        self.down1 = DownConv(4, 64)
        self.down2 = DownConv(46, 128)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)

        return x
