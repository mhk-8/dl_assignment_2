"""Training entrypoint
"""

import argparse
import os
import time
import copy
from typing import Dict, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import wandb

# ── Local Imports ──────────────────────────────────────────────────────────────
from data.pets_dataset import ( OxfordIIITPetDataset, get_train_transforms, get_val_transforms)
from models.classification import VGG11Classifier
from models.localization    import VGG11Localizer
from models.segmentation    import VGG11UNet
from models.multitask       import MultiTaskPerceptionModel
from losses.iou_loss         import IoULoss

# 1. UTILITIES & CONFIGURATION

def parse_args() -> argparse.Namespace:
    """Parses command line arguments for data, architecture, and training."""
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 Training")

    # Data Configuration
    p.add_argument("--data_root", type=str, required=True, help="Path to Oxford-IIIT Pet dataset")
    p.add_argument("--num_workers", type=int, default=2)

    # Task Selection
    p.add_argument("--task", type=str, default="classification", choices=["classification", "localization", "segmentation", "multitask"])

    # Architecture Hyperparameters
    p.add_argument("--num_classes", type=int, default=37)
    p.add_argument("--seg_classes", type=int, default=3)
    p.add_argument("--dropout_p", type=float, default=0.5, help="Prob for Section 2.2")
    p.add_argument("--freeze_encoder", type=str, default="none", choices=["none", "full", "partial"], help="Strategy for Section 2.3")

    # Optimization Parameters
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Multi-task Loss Scaling
    p.add_argument("--w_cls", type=float, default=1.0)
    p.add_argument("--w_loc", type=float, default=1.0)
    p.add_argument("--w_seg", type=float, default=1.0)

    # Checkpoint Management
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--cls_ckpt", type=str, default=None, help="Backbone initialization")
    p.add_argument("--loc_ckpt", type=str, default=None)
    p.add_argument("--seg_ckpt", type=str, default=None)

    # Logging (W&B)
    p.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--log_activation_hist", action="store_true", help="For Section 2.1")

    return p.parse_args()

# 2. MODEL BUILDING & WEIGHT INITIALIZATION

def _load_state(model: nn.Module, ckpt_path: str, device: torch.device):
    """Safely loads state dicts into the provided module."""
    payload = torch.load(ckpt_path, map_location=device)
    sd = payload.get("state_dict", payload)
    model.load_state_dict(sd, strict=False)
    print(f"    Loaded weights from {ckpt_path}")


def _apply_freeze(model: nn.Module, strategy: str):
    """Implements encoder freezing strategies for Section 2.3."""
    encoder = getattr(model, "encoder", None)
    if encoder is None or strategy == "none":
        return

    if strategy == "full":
        for p in encoder.parameters():
            p.requires_grad = False
        print("    Encoder: fully frozen")
    elif strategy == "partial":
        # Freeze early blocks (1-3), keep later blocks (4-5) trainable
        modules = [encoder.block1, encoder.pool1, encoder.block2, encoder.pool2, encoder.block3, encoder.pool3]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
        print("    Encoder: blocks 1-3 frozen, blocks 4-5 trainable")


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Factory function to instantiate models based on the selected task."""
    print(f"\nBuilding model for task: {args.task}")

    if args.task == "classification":
        model = VGG11Classifier(num_classes=args.num_classes, dropout_p=args.dropout_p)
        if args.cls_ckpt: _load_state(model, args.cls_ckpt, device)

    elif args.task == "localization":
        model = VGG11Localizer()
        if args.cls_ckpt: _load_state(model.encoder, args.cls_ckpt, device)
        _apply_freeze(model, args.freeze_encoder)

    elif args.task == "segmentation":
        model = VGG11UNet(num_classes=args.seg_classes)
        if args.cls_ckpt: _load_state(model.encoder, args.cls_ckpt, device)
        _apply_freeze(model, args.freeze_encoder)

    elif args.task == "multitask":
        model = MultiTaskPerceptionModel(
            num_breeds=args.num_classes, 
            seg_classes=args.seg_classes,
            dropout_p=args.dropout_p, 
            freeze_encoder=(args.freeze_encoder == "full")
        )
        # Load task-specific weights into the multitask heads
        if args.cls_ckpt: _load_state(model.encoder, args.cls_ckpt, device)
        if args.loc_ckpt:
            sd = torch.load(args.loc_ckpt, map_location=device).get("state_dict")
            model.bbox_head.load_state_dict({k.replace("regression_head.", ""): v for k, v in sd.items() if "regression_head" in k}, strict=False)
        if args.seg_ckpt:
            sd = torch.load(args.seg_ckpt, map_location=device).get("state_dict")
            model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("encoder.")}, strict=False)

    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable parameters: {trainable:,}")
    return model

# 3. METRIC CALCULATIONS

def compute_iou_batch(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Calculates per-sample IoU for [B, 4] normalized cx,cy,w,h tensors."""
    p_x1, p_y1 = pred_boxes[:, 0] - pred_boxes[:, 2]/2, pred_boxes[:, 1] - pred_boxes[:, 3]/2
    p_x2, p_y2 = pred_boxes[:, 0] + pred_boxes[:, 2]/2, pred_boxes[:, 1] + pred_boxes[:, 3]/2
    t_x1, t_y1 = target_boxes[:, 0] - target_boxes[:, 2]/2, target_boxes[:, 1] - target_boxes[:, 3]/2
    t_x2, t_y2 = target_boxes[:, 0] + target_boxes[:, 2]/2, target_boxes[:, 1] + target_boxes[:, 3]/2

    i_x1, i_y1 = torch.max(p_x1, t_x1), torch.max(p_y1, t_y1)
    i_x2, i_y2 = torch.min(p_x2, t_x2), torch.min(p_y2, t_y2)
    
    inter = torch.clamp(i_x2 - i_x1, 0) * torch.clamp(i_y2 - i_y1, 0)
    union = (pred_boxes[:, 2] * pred_boxes[:, 3]) + (target_boxes[:, 2] * target_boxes[:, 3]) - inter + eps
    return (inter + eps) / (union + eps)


def compute_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor, num_classes: int = 3, eps: float = 1e-6) -> float:
    """Computes Mean Dice score across classes."""
    pred = pred_mask.argmax(dim=1)
    dice_scores = []
    for c in range(num_classes):
        p, t = (pred == c).float(), (true_mask == c).float()
        intersection = (p * t).sum()
        dice_scores.append((2 * intersection + eps) / (p.sum() + t.sum() + eps))
    return float(torch.stack(dice_scores).mean())

# 4. TRAINING & VALIDATION ENGINES

def train_epoch(model, loader, optimizer, cls_crit, loc_crit, seg_crit, args, device, epoch) -> dict:
    model.train()
    task = args.task
    total_l, c_l, l_l, s_l = 0.0, 0.0, 0.0, 0.0

    for idx, batch in enumerate(loader):
        img, lbl, box, msk = batch["image"].to(device), batch["label"].to(device), batch["bbox"].to(device), batch["mask"].to(device)
        optimizer.zero_grad()

        if task == "classification":
            loss = cls_crit(model(img), lbl)
            c_l += loss.item()
        elif task == "localization":
            loss = loc_crit(model(img), box)
            l_l += loss.item()
        elif task == "segmentation":
            loss = seg_crit(model(img), msk)
            s_l += loss.item()
        elif task == "multitask":
            out = model(img)
            lc, ll, ls = cls_crit(out["classification"], lbl), loc_crit(out["localization"], box), seg_crit(out["segmentation"], msk)
            loss = (args.w_cls * lc) + (args.w_loc * ll) + (args.w_seg * ls)
            c_l, l_l, s_l = c_l + lc.item(), l_l + ll.item(), s_l + ls.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_l += loss.item()

    # Consolidate metrics
    n = len(loader)
    res = {"train/loss": total_l / n}
    if task in ("classification", "multitask"): res["train/cls_loss"] = c_l / n
    if task in ("localization", "multitask"): res["train/loc_loss"] = l_l / n
    if task in ("segmentation", "multitask"): res["train/seg_loss"] = s_l / n
    return res


def val_epoch(model, loader, cls_crit, loc_crit, seg_crit, args, device, epoch) -> dict:
    model.eval()
    total_l, all_lbl, all_prd, ious, dices, accs = 0.0, [], [], [], [], []
    seg_samples = []

    with torch.no_grad():
        for batch in loader:
            img, lbl, box, msk = batch["image"].to(device), batch["label"].to(device), batch["bbox"].to(device), batch["mask"].to(device)

            if args.task == "classification":
                out = model(img)
                total_l += cls_crit(out, lbl).item()
                all_prd.extend(out.argmax(1).cpu().tolist()); all_lbl.extend(lbl.cpu().tolist())
            elif args.task == "localization":
                out = model(img)
                total_l += loc_crit(out, box).item()
                ious.extend(compute_iou_batch(out, box).cpu().tolist())
            elif args.task == "segmentation":
                out = model(img)
                total_l += seg_crit(out, msk).item()
                dices.append(compute_dice(out, msk)); accs.append((out.argmax(1) == msk).float().mean().item())
                if len(seg_samples) < 5: seg_samples.append({"image": img[0].cpu(), "gt_mask": msk[0].cpu(), "pred_mask": out[0].argmax(0).cpu()})
            elif args.task == "multitask":
                out = model(img)
                total_l += (args.w_cls * cls_crit(out["classification"], lbl) + args.w_loc * loc_crit(out["localization"], box) + args.w_seg * seg_crit(out["segmentation"], msk)).item()
                all_prd.extend(out["classification"].argmax(1).cpu().tolist()); all_lbl.extend(lbl.cpu().tolist())
                ious.extend(compute_iou_batch(out["localization"], box).cpu().tolist())
                dices.append(compute_dice(out["segmentation"], msk)); accs.append((out["segmentation"].argmax(1) == msk).float().mean().item())
                if len(seg_samples) < 5: seg_samples.append({"image": img[0].cpu(), "gt_mask": msk[0].cpu(), "pred_mask": out["segmentation"][0].argmax(0).cpu()})

    res = {"val/loss": total_l / len(loader)}
    if all_lbl: res["val/macro_f1"] = f1_score(all_lbl, all_prd, average="macro", zero_division=0)
    if ious: res["val/mean_iou"] = float(np.mean(ious))
    if dices: res["val/dice_score"], res["val/pixel_accuracy"] = float(np.mean(dices)), float(np.mean(accs))
    
    # Section 2.6 Visuals
    if seg_samples and (epoch % 5 == 0 or epoch == 1): _log_seg_samples(seg_samples, epoch)
    return res


# 5. W&B HOOKS & VISUALIZATION HELPERS

class ActivationHook:
    """Section 2.1 — Captures layer outputs for histogram logging."""
    def __init__(self): self.activation, self._handle = None, None
    def register(self, layer: nn.Module): self._handle = layer.register_forward_hook(self._hook)
    def _hook(self, m, i, o): self.activation = o.detach().cpu()
    def remove(self): 
        if self._handle: self._handle.remove()

_SEG_PALETTE = np.array([[255, 255, 255], [0, 0, 0], [128, 128, 128]], dtype=np.uint8)

def _denorm_image(t: torch.Tensor) -> np.ndarray:
    """Section 2.4 — Reverses ImageNet normalization."""
    img = (t.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    return np.clip(img, 0, 255).astype(np.uint8)

def _log_seg_samples(samples: list, epoch: int):
    """Section 2.6 — Logs side-by-side mask comparisons."""
    wb_imgs = []
    for s in samples:
        row = np.concatenate([_denorm_image(s["image"]), _SEG_PALETTE[np.clip(s["gt_mask"].numpy(), 0, 2)], _SEG_PALETTE[np.clip(s["pred_mask"].numpy(), 0, 2)]], axis=1)
        wb_imgs.append(wandb.Image(row, caption=f"Epoch {epoch} | Orig | GT | Pred"))
    wandb.log({"seg/sample_predictions": wb_imgs}, commit=False)


# 6. MAIN EXECUTION

def main():
    args, device = parse_args(), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(project=args.wandb_project, name=args.run_name or f"{args.task}_dp{args.dropout_p}", config=vars(args))
    train_loader, val_loader = DataLoader(OxfordIIITPetDataset(args.data_root, "train", get_train_transforms()), args.batch_size, True, num_workers=args.num_workers), DataLoader(OxfordIIITPetDataset(args.data_root, "val", get_val_transforms()), args.batch_size, False, num_workers=args.num_workers)
    
    model = build_model(args, device)
    cls_c, loc_c, seg_c = nn.CrossEntropyLoss().to(device), IoULoss(reduction="mean").to(device), nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_m, m_key = 0.0, {"classification": "val/macro_f1", "localization": "val/mean_iou", "segmentation": "val/dice_score", "multitask": "val/dice_score"}[args.task]

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        start = time.time()
        
        t_metrics = train_epoch(model, train_loader, optimizer, cls_c, loc_c, seg_c, args, device, epoch)
        v_metrics = val_epoch(model, val_loader, cls_c, loc_c, seg_c, args, device, epoch)
        scheduler.step()

        # Section 2.1 Logging
        if args.log_activation_hist:
            h = ActivationHook(); h.register(getattr(model, "encoder", model).block2[0])
            model.eval(); model(next(iter(val_loader))["image"].to(device)); h.remove()
            wandb.log({"activations/block2_conv_hist": wandb.Histogram(h.activation.flatten().numpy())}, commit=False)

        logs = {**t_metrics, **v_metrics, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "epoch_time": time.time() - start}
        wandb.log(logs)
        print(" | ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float)))

        if v_metrics.get(m_key, 0.0) > best_m:
            best_m = v_metrics[m_key]
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "best_metric": best_m}, os.path.join(args.ckpt_dir, f"{args.task}.pth"))

    wandb.finish()

if __name__ == "__main__":
    main()