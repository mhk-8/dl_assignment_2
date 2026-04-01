"""Segmentation model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder, VGG11
from .layers import CustomDropout, conv_block
    
class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        # Contracting path (encoder)
        self.encoder = VGG11(in_channels=in_channels)

        # Expansive path (decoder) — mirrors encoder resolution stages
        # bottleneck p5: [B, 512,   7,   7]
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = conv_block(512 + 512, 512)   # up + block4 skip [B,512,14,14]

        # [B, 512, 14, 14]
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = conv_block(256 + 256, 256)   # up + block3 skip [B,256,28,28]

        # [B, 256, 28, 28]
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(128 + 128, 128)   # up + block2 skip [B,128,56,56]

        # [B, 128, 56, 56]
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(64 + 64, 64)      # up + block1 skip [B,64,112,112]

        # [B, 64, 112, 112] — final upsample to 224×224, no skip
        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 32)

        # 1×1 projection to class logits
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Extract features from VGG11 backbone
        bottleneck, features = self.encoder(x, return_features=True)

        # Expansive path with skip connections (Feature Fusion) 
        d5 = self.dec5(torch.cat([self.up5(bottleneck), features["block4"]], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5),         features["block3"]], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4),         features["block2"]], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3),         features["block1"]], dim=1))
        d1 = self.dec1(self.up1(d2))

        return self.final_conv(d1)
