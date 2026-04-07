"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder, VGG11

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
        
                
        # Regression Head
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            
            # Layer 1
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            
            # Output Layer: 4 coordinates [xc, yc, w, h]
            nn.Linear(512, 4),
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
        # Pass through the regression head
        bboxes = self.regression_head(features)
        
        return bboxes
