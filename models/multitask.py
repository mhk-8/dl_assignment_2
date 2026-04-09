"""Unified multi-task model
"""
import os
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder, VGG11
from .layers import CustomDropout, conv_block

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path:str = "checkpoints/classifier.pth", localizer_path:str = "checkpoints/localizer.pth", unet_path:str = "checkpoints/unet.pth"):
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
        import gdown
        gdown.download(id="1DB83O3KYIu0TWa2FCB7unfB6ge-8-gUY", output=classifier_path, quiet=False)
        gdown.download(id="10kmNpPlXswYe0faWasSIA9rWXxLcS-GG", output=localizer_path, quiet=False)
        gdown.download(id="1JlzkeV8cRjSfJtnpgoDwyaiEiJHeaH0Y", output=unet_path, quiet=False)
        # Shared Backbone
        self.encoder = VGG11(in_channels=in_channels)
                
        # Classification Head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5), # Regularization effect to be analyzed in W&B 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
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
            nn.Sigmoid() 
        )

        # Segmentation Decoder (from Task 3)
        # We reuse the logic from VGG11UNet expansive path
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = conv_block(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = conv_block(512+256 , 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256+128 , 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128+64 , 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(32 +64, 32)
        self.final_seg = nn.Conv2d(32, seg_classes, kernel_size=1)
        
        #load pretrained weights
        device = torch.device("cpu")
        self._load_classifier(classifier_path, device)
        self._load_localizer(localizer_path,   device)
        self._load_unet(unet_path,             device)
    
    #  Weight loading helpers

    def _get_sd(self, path: str, device):
        """Load state dict from path if it exists."""
        if not os.path.exists(path):
            print(f"  Warning: checkpoint not found: {path}")
            return None
        payload = torch.load(path, map_location=device)
        return payload.get("state_dict", payload)

    def _load_classifier(self, path: str, device):
        sd = self._get_sd(path, device)
        if sd is None:
            return
        enc_sd = {k.replace("encoder.", ""): v
                  for k, v in sd.items() if k.startswith("encoder.")}
        cls_sd = {k.replace("classifier.", ""): v
                  for k, v in sd.items() if k.startswith("classifier.")}
        self.encoder.load_state_dict(enc_sd, strict=False)
        self.cls_head.load_state_dict(cls_sd, strict=False)

    def _load_localizer(self, path: str, device):
        sd = self._get_sd(path, device)
        if sd is None:
            return
        bbox_sd = {k.replace("regression_head.", ""): v
                   for k, v in sd.items() if "regression_head" in k}
        self.bbox_head.load_state_dict(bbox_sd, strict=False)

    def _load_unet(self, path: str, device):
        sd = self._get_sd(path, device)
        if sd is None:
            return
        dec_sd = {k: v for k, v in sd.items()
                  if not k.startswith("encoder.")}
        self.load_state_dict(dec_sd, strict=False)
        
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
        bbox_out = self.bbox_head(bottleneck) * 224.0

        # Forward pass using all 5 skip connections
        d5 = self.dec5(torch.cat([self.up5(bottleneck), skips["block5"]], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), skips["block4"]], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), skips["block3"]], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skips["block2"]], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), skips["block1"]], dim=1))
        seg_out = self.final_seg(d1)

        return {
            "classification": cls_out,
            "localization": bbox_out,
            "segmentation": seg_out,
        }
