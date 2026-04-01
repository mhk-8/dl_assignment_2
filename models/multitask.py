"""Unified multi-task model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout, conv_block

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth", dropout_p: float = 0.5, freeze_encoder: bool = False):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        # Shared Backbone
        self.encoder = VGG11Encoder(in_channels=in_channels)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Classification Head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p), # Regularization effect to be analyzed in W&B 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds)
        )

        # Localization Head 
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.Sigmoid() # Constrains to normalized coordinate space 
        )

        # Segmentation Decoder (from Task 3)
        # We reuse the logic from VGG11UNet expansive path
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = self._conv_block(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(32, 32)
        self.final_seg = nn.Conv2d(32, seg_classes, kernel_size=1)
        
        
    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
     
        # Single Forward Pass
        bottleneck, skips = self.encoder(x, return_features=True)

        # Classification 
        cls_out = self.cls_head(bottleneck)

        # Localization 
        bbox_out = self.bbox_head(bottleneck)

        # Segmentation 
        d5 = self.dec5(torch.cat([self.up5(bottleneck), skips["block4"]], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5),         skips["block3"]], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4),         skips["block2"]], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3),         skips["block1"]], dim=1))
        d1 = self.dec1(self.up1(d2))
        seg_out = self.final_seg(d1)

        return {
            "classification": cls_out,
            "localization": bbox_out,
            "segmentation": seg_out,
        }
