"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        # validating p value should be [0,1)
        if not (0 <= p < 1):
            raise ValueError(f"Dropout probability must be with in range [0,1)")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        if not self.training or self.p == 0:
            return x
        # Create the binary mask
        # Generate a tensor of the same shape as x with values from 0 to 1.
        # Elements > p become 1 (keep), elements <= p become 0 (drop).
        mask = (torch.rand(x.shape, device=x.device) > self.p).to(dtype=x.dtype)
        
        # Apply mask and Inverted Scaling
        # We divide by (1 - p) so the activations don't drop in magnitude during training.
        return (x * mask) / (1 - self.p)

def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv → BN → ReLU refinement block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
