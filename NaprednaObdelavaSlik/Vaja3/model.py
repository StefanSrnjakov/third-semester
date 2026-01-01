import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Down path
        self.down1 = DownBlock(3, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        
        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bottleneck_bn1 = nn.BatchNorm2d(256)
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bottleneck_bn2 = nn.BatchNorm2d(256)
        self.bottleneck_relu2 = nn.ReLU(inplace=True)
        
        # Up path
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        
        # Final layer (method 3: no sigmoid, BCEWithLogitsLoss)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        # Down path
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        
        # Bottleneck
        x = self.bottleneck_relu1(self.bottleneck_bn1(self.bottleneck_conv1(x)))
        x = self.bottleneck_relu2(self.bottleneck_bn2(self.bottleneck_conv2(x)))
        
        # Up path with skip connections
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        
        # Final layer (no activation - BCEWithLogitsLoss handles it)
        x = self.final_conv(x)
        
        return x

