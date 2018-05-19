import torch.nn as N
import torch.nn.functional as F
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
                                padding=1,
                                stride=1)

        self.conv_out = N.Conv3d(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 padding=1,
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
        self.up_conv = N.Upsample(scale_factor=2)

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
        self.max_pool = N.MaxPool3d(kernel_size=2)

    def pad(self, x, prev):

        x_size = x.size()
        prev_size = prev.size()

        x_depth = x_size[2]
        x_height = x_size[3]
        x_width = x_size[4]

        prev_depth = prev_size[2]
        prev_height = prev_size[3]
        prev_width = prev_size[4]

        diff_depth = x_depth - prev_depth
        depth_pad1 = diff_depth // 2
        depth_pad2 = diff_depth - depth_pad1

        diff_height = x_height - prev_height
        height_pad1 = diff_height // 2
        height_pad2 = diff_height - height_pad1

        diff_width = x_width - prev_width
        width_pad1 = diff_width // 2
        width_pad2 = diff_width - width_pad1

        padding = (width_pad1, width_pad2,
                   height_pad1, height_pad2,
                   depth_pad1, depth_pad2)

        prev = F.pad(input=prev, pad=padding, mode="constant", value=0)
        print(f"After padding prev {prev.size()}")
        print(f"After padding x {x.size()}")

    def forward(self, x, prev):
        x = self.up_conv(x)


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
        self.down1 = DownConv(4, 8)
        self.down2 = DownConv(8, 16)
        self.down3 = DownConv(16, 32)
        self.down4 = DownConv(32, 64)

        self.middle1 = N.Conv3d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                padding=1,
                                stride=1)

        self.middle2 = N.Conv3d(in_channels=128,
                                out_channels=128,
                                kernel_size=3,
                                padding=1,
                                stride=1)

        self.upconv4 = UpConv(128, 64)

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.debug("U-Net initialised")

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        print(f"After 2 down convs {x2.size()}")
        x3 = self.down3(x2)
        print(f"After 3 down convs {x3.size()}")
        x4 = self.down4(x3)
        print(f"After all down convs {x4.size()}")

        xm = self.middle1(x4)
        print(f"After first middle {xm.size()}")
        xm = self.middle2(xm)
        print(f"After second middle {xm.size()}")

        xup4 = self.upconv4(xm, x4)
        #xup3 = self.upconv3(xup4, x3)
        #xup2 = self.upconv2(x, x2)
        #xup1 = self.upconv1(xm, x1)

        return xup4
