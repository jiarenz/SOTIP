""" https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py """


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, batch_norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if batch_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(64, mid_channels),
                # nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                # nn.GroupNorm(64, out_channels),
                # # nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                # nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     DoubleConv(in_channels, out_channels)
        # )
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        # ks = 3
        # self.stride_2_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, ks, padding=ks//2, stride=2),
        #     nn.ReLU(inplace=True),
        #     DoubleConv(out_channels, out_channels)
        # )

    def forward(self, x):
        return self.maxpool_conv(x)
        # return self.stride_2_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = nn.ReLU()(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return nn.ReLU()(self.conv(x))
        # return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)