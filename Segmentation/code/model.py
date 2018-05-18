import torch.nn as N
import logging


class DownConv(N.Module):

    """
        Input(Batch, Channel, Depth, Height, Width) => Conv() => Relu * 2 => MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = in_channels
        self.outc = out_channels

        self.conv_in = N.Conv3d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=0,
                                stride=1)

        self.conv_out = N.Conv3d(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 padding=0,
                                 stride=1)

        self.relu = N.ReLU()
        self.max_pool = N.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_out(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class UpConv(N.Module):

    """
        Input(Batch, Channel, Depth, Height, Width) => Conv() => Relu * 2 => MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = in_channels
        self.outc = out_channels

        self.up_conv = N.Upsample(scale_factor=2)

        self.conv_in = N.Conv3d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=0,
                                stride=1)

        self.conv_out = N.Conv3d(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 padding=0,
                                 stride=1)

        self.relu = N.ReLU()
        self.max_pool = N.MaxPool3d(kernel_size=2)

    def forward(self, x, prev):
        x = self.up_conv(x)
        print(x.size())
        print(prev.size())
        # x = self.conv_in(x)
        # x = self.conv_out(x)
        # x = self.relu(x)
        # x = self.max_pool(x)
        return x


class UNet(N.Module):

    """
        Defines U-Net using conv and upconv blocks defined above.
    """

    def __init__(self):
        super().__init__()
        self.down1 = DownConv(4, 64)
        self.down2 = DownConv(self.down1.outc, self.down1.outc * 2)
        self.down3 = DownConv(self.down2.outc, self.down2.outc * 2)
        self.down4 = DownConv(self.down3.outc, self.down3.outc * 2)

        self.middle1 = N.Conv3d(in_channels=self.down4.outc,
                                out_channels=self.down4.outc * 2,
                                kernel_size=3,
                                stride=1)

        self.middle2 = N.Conv3d(in_channels=self.down4.outc * 2,
                                out_channels=self.down4.outc * 2,
                                kernel_size=3,
                                stride=1)

        self.upconv4 = UpConv(1024, 512)

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.debug("U-Net initialised")

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        xm = self.middle1(x2)
        xm = self.middle2(xm)

        #xup4 = self.upconv4(xm, x4)
        #xup3 = self.upconv3(xup4, x3)
        #xup2 = self.upconv2(x, x2)
        #xup1 = self.upconv1(xm, x1)

        return xm
