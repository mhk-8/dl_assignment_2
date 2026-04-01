"""Classification components
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        # Instantiate backbone encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # Define the classification head
        # We use AdaptiveAvgPool2d to ensure the flattened size is always 25088
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            
            # Layer 1
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            
            # Layer 2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            
            # Output Layer: num_classes (37 pet breeds)
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        
        # Pass through the encoder 
        # We do not need intermediate features here (return_features=False)
        features = self.encoder(x)
        # Pass bottleneck features through the MLP head
        logits = self.classifier(features)
        
        return logits
