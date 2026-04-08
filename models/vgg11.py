"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, use_bn: bool = True):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # Block 1: 64 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            *(([nn.BatchNorm2d(64)]) if use_bn else []),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 128 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            *(([nn.BatchNorm2d(128)]) if use_bn else []),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 256 filters (2 conv layers)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            *(([nn.BatchNorm2d(256)]) if use_bn else []),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            *(([nn.BatchNorm2d(256)]) if use_bn else []),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 512 filters (2 conv layers)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            *(([nn.BatchNorm2d(512)]) if use_bn else []),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            *(([nn.BatchNorm2d(512)]) if use_bn else []),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 512 filters (2 conv layers)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            *(([nn.BatchNorm2d(512)]) if use_bn else []),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            *(([nn.BatchNorm2d(512)]) if use_bn else []),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, return_features: bool = False ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        
        # Pass through blocks and capture pre-pool features for U-Net skip connections
        s1 = self.block1(x)
        p1 = self.pool1(s1)

        s2 = self.block2(p1)
        p2 = self.pool2(s2)

        s3 = self.block3(p2)
        p3 = self.pool3(s3)

        s4 = self.block4(p3)
        p4 = self.pool4(s4)

        s5 = self.block5(p4)
        p5 = self.pool5(s5)

        if return_features:
            features = {
                "block1": s1,  
                "block2": s2,  
                "block3": s3,  
                "block4": s4,  
                "block5": s5  
            }
            return p5, features
        
        return p5
    
# Alias so autograder can do: from models.vgg11 import VGG11
VGG11 = VGG11Encoder