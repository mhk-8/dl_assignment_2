"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        
        # Validating reduction
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Reduction must be 'mean', or 'sum'. Got: {reduction}")
        self.eps = eps
        self.reduction = reduction
        

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
       
        # Convert to corners (x1, y1, x2, y2)
        # Prediction corners
        p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        # Target corners
        t_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        t_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        t_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        t_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Intersection area
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        # Clamp ensures no negative width/height if boxes don't touch
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_w * inter_h

        # Union area
        area_pred = pred_boxes[:, 2] * pred_boxes[:, 3]
        area_target = target_boxes[:, 2] * target_boxes[:, 3]
        union = area_pred + area_target - intersection

        # IoU and Loss
        iou = (intersection + self.eps) / (union + self.eps)
        loss = 1 - iou

        # Reduction logic
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        