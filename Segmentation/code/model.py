import torch
import torch.nn as N
import torch.nn.functional as F
import logging


class DownConv(N.Module):

    """
        Input(Batch, Channel, Depth, Height, Width) => (Conv() => Relu) * 2 =>
        MaxPool
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
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class UpConv(N.Module):

    """
        Previous \\
        Current =====> input => (Conv3d => ReLU) * 2
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

    def pad_and_concat(self, x, prev):

        """
            Matches the dimensions of prev and x and then concatenates.
        """

        x_size = x.size()
        prev_size = prev.size()

        x_depth = x_size[2]
        x_height = x_size[3]
        x_width = x_size[4]
        x_channels = x_size[1]

        prev_depth = prev_size[2]
        prev_height = prev_size[3]
        prev_width = prev_size[4]
        prev_channels = prev_size[1]

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
        x = x[:, :int(x_channels - prev_channels), :, :, :]
        return torch.cat((x, prev), dim=1)

    def forward(self, x, prev):
        x = self.up_conv(x)
        x = self.pad_and_concat(x, prev)
        x = self.conv_in(x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
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
                                out_channels=124,
                                kernel_size=3,
                                padding=1,
                                stride=1)

        self.middle2 = N.Conv3d(in_channels=124,
                                out_channels=124,
                                kernel_size=3,
                                padding=1,
                                stride=1)

        self.upconv4 = UpConv(124, 64)
        self.upconv3 = UpConv(64, 32)
        self.upconv2 = UpConv(32, 16)
        self.upconv1 = UpConv(16, 8)

        self.classify_conv = N.Conv3d(in_channels=8,
                                      out_channels=4,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1)

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.debug("U-Net initialised")

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        xm = self.middle1(x4)
        xm = self.middle2(xm)

        x = self.upconv4(xm, x4)
        x = self.upconv3(x, x3)
        x = self.upconv2(x, x2)
        x = self.upconv1(x, x1)

        x = self.classify_conv(x)

        return x


class LameCNN(N.Module):
    """
        Really small FCNN to check if memory problems are not due to my
        breath takingly poor coding abilities
    """

    def __init__(self):
        super().__init__()
        self.conv = N.Conv3d(in_channels=4,
                             out_channels=1,
                             kernel_size=3,
                             padding=1,
                             stride=1)

        self.end_conv = N.Conv3d(in_channels=1,
                                 out_channels=4,
                                 kernel_size=3,
                                 padding=1,
                                 stride=1)

        self.max_pool = N.MaxPool3d(kernel_size=2)
        self.relu = N.ReLU()
        self.up_conv = N.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.up_conv(x)
        x = self.end_conv(x)
        return x
