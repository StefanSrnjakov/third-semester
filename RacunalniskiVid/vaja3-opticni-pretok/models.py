"""
Neural network models for optical flow estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip


class FlowNetSimple(nn.Module):
    def __init__(
        self,
        input_channels=6,
        down_ch1=16,
        down_ch2=32,
        down_ch3=64,
        bottleneck_ch=128,
        flow_channels=2
    ):
        super().__init__()

        self.down1 = DownBlock(input_channels, down_ch1, kernel_size=7)
        self.down2 = DownBlock(down_ch1, down_ch2, kernel_size=5)
        self.down3 = DownBlock(down_ch2, down_ch3, kernel_size=3)

        self.conv4_1 = nn.Conv2d(down_ch3, bottleneck_ch, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(bottleneck_ch)
        self.conv4_2 = nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(bottleneck_ch)

        self.up3 = nn.ConvTranspose2d(bottleneck_ch, down_ch3, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(down_ch3 + down_ch3, down_ch3, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(down_ch3, down_ch2, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(down_ch2 + down_ch2, down_ch2, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose2d(down_ch2, down_ch1, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(down_ch1 + down_ch1, down_ch1, kernel_size=3, padding=1)

        self.flow_pred = nn.Conv2d(down_ch1, flow_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)

        # Bottleneck
        x4 = F.relu(self.bn4_1(self.conv4_1(x3)))
        x4 = F.relu(self.bn4_2(self.conv4_2(x4)))

        # Decoder with skip connections
        u3 = self.up3(x4)
        u3 = torch.cat([u3, skip3], dim=1)
        u3 = F.relu(self.dec3(u3))

        u2 = self.up2(u3)
        u2 = torch.cat([u2, skip2], dim=1)
        u2 = F.relu(self.dec2(u2))

        u1 = self.up1(u2)
        u1 = torch.cat([u1, skip1], dim=1)
        u1 = F.relu(self.dec1(u1))

        # Flow prediction
        flow = self.flow_pred(u1)
        return flow

