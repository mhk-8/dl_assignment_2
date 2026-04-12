"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder, VGG11
from .layers import CustomDropout

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""
    # Outputs [xc, yc, w, h] in pixel space (0-224).
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        # Reuse the VGG11 Encoder
        self.encoder = VGG11(in_channels=in_channels)
        
                
        self.layer1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(start_dim=1),
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(4096, 1024, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 4, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        
        # Extract features (skip return_features for now)
        features = self.encoder(x)
        x1 = self.layer1(features)
        x2 = self.layer2(x1)
        bboxes = self.layer3(x2) * 224.0
        
        return bboxes
