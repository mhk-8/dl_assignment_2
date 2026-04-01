"""Inference and evaluation
"""
import argparse
import os
import copy
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import wandb
from PIL import Image as PILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset, get_val_transforms
from models.multitask import MultiTaskPerceptionModel
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

# 1. ARGUMENT PARSING

def parse_args() -> argparse.Namespace:
    """Parses command line arguments for evaluation configuration."""
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 Inference")

    # Data Path
    p.add_argument("--data_root", type=str, required=True, help="Path to Oxford-IIIT Pet dataset")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)

    # Mandatory Checkpoints
    p.add_argument("--cls_ckpt", type=str, default="checkpoints/classifier.pth")
    p.add_argument("--loc_ckpt", type=str, default="checkpoints/localizer.pth")
    p.add_argument("--seg_ckpt", type=str, default="checkpoints/unet.pth")

    # Architecture Settings
    p.add_argument("--num_classes", type=int, default=37)
    p.add_argument("--seg_classes", type=int, default=3)
    p.add_argument("--dropout_p", type=float, default=0.5)

    # Logging & Mode
    p.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--mode", type=str, default="multitask", choices=["multitask", "classification", "localization", "segmentation"])

    return p.parse_args()

# 2. IMAGE PROCESSING HELPERS

def _denorm_image(t: torch.Tensor) -> np.ndarray:
    """Reverses ImageNet normalization: [3, H, W] tensor -> [H, W, 3] uint8."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (t.permute(1, 2, 0).numpy() * std + mean) * 255
    return np.clip(img, 0, 255).astype(np.uint8)

# Color palette for Trimap: 0=FG (White), 1=BG (Black), 2=Boundary (Grey)
_SEG_PALETTE = np.array([[255, 255, 255], [0, 0, 0], [128, 128, 128]], dtype=np.uint8)

def _mask_to_rgb(mask: torch.Tensor) -> np.ndarray:
    """Maps segmentation indices to RGB colors."""
    m = np.clip(mask.numpy().astype(np.int64), 0, 2)
    return _SEG_PALETTE[m]

def _draw_box(img: np.ndarray, box: np.ndarray, color: tuple, W: int, H: int, thickness: int = 2):
    """Draws normalized (cx, cy, w, h) boxes on an image in-place."""
    cx, cy, bw, bh = box
    x1, y1 = int((cx - bw/2) * W), int((cy - bh/2) * H)
    x2, y2 = int((cx + bw/2) * W), int((cy + bh/2) * H)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2)
    img[y1:y1+thickness, x1:x2] = color
    img[y2:y2+thickness, x1:x2] = color
    img[y1:y2, x1:x1+thickness] = color
    img[y1:y2, x2:x2+thickness] = color

# 3. QUANTITATIVE METRIC CALCULATIONS

def compute_iou_per_sample(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Calculates IoU for Task 1.2 and Task 2.5 evaluation"""
    p_x1, p_y1 = pred_boxes[:, 0] - pred_boxes[:, 2]/2, pred_boxes[:, 1] - pred_boxes[:, 3]/2
    p_x2, p_y2 = pred_boxes[:, 0] + pred_boxes[:, 2]/2, pred_boxes[:, 1] + pred_boxes[:, 3]/2
    t_x1, t_y1 = target_boxes[:, 0] - target_boxes[:, 2]/2, target_boxes[:, 1] - target_boxes[:, 3]/2
    t_x2, t_y2 = target_boxes[:, 0] + target_boxes[:, 2]/2, target_boxes[:, 1] + target_boxes[:, 3]/2

    i_x1, i_y1 = torch.max(p_x1, t_x1), torch.max(p_y1, t_y1)
    i_x2, i_y2 = torch.min(p_x2, t_x2), torch.min(p_y2, t_y2)

    inter = torch.clamp(i_x2 - i_x1, 0) * torch.clamp(i_y2 - i_y1, 0)
    union = (pred_boxes[:, 2] * pred_boxes[:, 3]) + (target_boxes[:, 2] * target_boxes[:, 3]) - inter + eps
    return (inter + eps) / (union + eps)


def compute_map(all_iou: list) -> float:
    """Approximates Mean Average Precision (mAP) for Section 2.8"""
    thresholds = np.arange(0.5, 1.0, 0.05)
    iou_arr = np.array(all_iou)
    aps = [(iou_arr >= thr).sum() / len(iou_arr) if len(iou_arr) > 0 else 0.0 for thr in thresholds]
    return float(np.mean(aps))

# 4. MODEL LOADING & EVALUATION LOGIC

def build_eval_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Instantiates the model and loads task checkpoints"""
    print(f"\nBuilding evaluation model: {args.mode}")

    if args.mode == "multitask":
        model = MultiTaskPerceptionModel(num_breeds=args.num_classes, seg_classes=args.seg_classes, dropout_p=args.dropout_p)
        # Unified loading logic
        for ckpt, module_attr in [(args.cls_ckpt, "cls_head"), (args.loc_ckpt, "bbox_head"), (args.seg_ckpt, None)]:
            if not os.path.exists(ckpt): continue
            sd = torch.load(ckpt, map_location=device).get("state_dict")
            if module_attr == "cls_head":
                model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}, strict=False)
                model.cls_head.load_state_dict({k.replace("classifier.", ""): v for k, v in sd.items() if k.startswith("classifier.")}, strict=False)
            elif module_attr == "bbox_head":
                model.bbox_head.load_state_dict({k.replace("regression_head.", ""): v for k, v in sd.items() if "regression_head" in k}, strict=False)
            else:
                model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("encoder.")}, strict=False)
            print(f"    Loaded components from {ckpt}")
    else:
        # Single-task loading logic
        model_cls = {"classification": VGG11Classifier, "localization": VGG11Localizer, "segmentation": VGG11UNet}[args.mode]
        model = model_cls(num_classes=args.num_classes if args.mode != "localization" else None)
        ckpt_path = {"classification": args.cls_ckpt, "localization": args.loc_ckpt, "segmentation": args.seg_ckpt}[args.mode]
        model.load_state_dict(torch.load(ckpt_path, map_location=device).get("state_dict", {}), strict=False)

    return model.to(device).eval()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, args: argparse.Namespace, device: torch.device):
    """Main evaluation loop aggregating metrics and visual records"""
    all_lbl, all_prd, all_iou, all_dice, all_acc = [], [], [], [], []
    seg_samples, bbox_records = [], []

    for batch in loader:
        img, lbl, box, msk = batch["image"].to(device), batch["label"].to(device), batch["bbox"].to(device), batch["mask"].to(device)
        out = model(img)

        # Unified unpacking based on mode
        c_out = out["classification"] if isinstance(out, dict) else (out if args.mode == "classification" else None)
        l_out = out["localization"] if isinstance(out, dict) else (out if args.mode == "localization" else None)
        s_out = out["segmentation"] if isinstance(out, dict) else (out if args.mode == "segmentation" else None)

        if c_out is not None:
            all_prd.extend(c_out.argmax(1).cpu().tolist()); all_lbl.extend(lbl.cpu().tolist())
        if l_out is not None:
            ious = compute_iou_per_sample(l_out, box)
            all_iou.extend(ious.cpu().tolist())
            if len(bbox_records) < 15:
                for i in range(len(img)):
                    if len(bbox_records) >= 15: break
                    bbox_records.append({"image": img[i].cpu(), "gt_box": box[i].cpu().numpy(), "pred_box": l_out[i].cpu().numpy(), "iou": float(ious[i].cpu())})
        if s_out is not None:
            # Mask metrics 
            pred_m = s_out.argmax(1)
            all_dice.append(float(np.mean([((pred_m == c).float() * (msk == c).float()).sum() * 2 / ((pred_m == c).float().sum() + (msk == c).float().sum() + 1e-6) for c in range(args.seg_classes)])))
            all_acc.append(float((pred_m == msk).float().mean()))
            if len(seg_samples) < 5:
                seg_samples.append({"image": img[0].cpu(), "gt_mask": msk[0].cpu(), "pred_mask": pred_m[0].cpu()})

    res = {}
    if all_lbl: res["test/macro_f1"] = f1_score(all_lbl, all_prd, average="macro", zero_division=0)
    if all_iou: res["test/mean_iou"], res["test/mAP"] = float(np.mean(all_iou)), compute_map(all_iou)
    if all_dice: res["test/dice_score"], res["test/pixel_accuracy"] = float(np.mean(all_dice)), float(np.mean(all_acc))

    return res, seg_samples, bbox_records

# 5. MAIN EXECUTION

def main():
    args, device = parse_args(), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, name=f"eval_{args.mode}", config=vars(args), job_type="eval")

    test_loader = DataLoader(OxfordIIITPetDataset(args.data_root, "test", get_val_transforms()), args.batch_size, False, num_workers=args.num_workers)
    model = build_eval_model(args, device)

    print("\nRunning test evaluation...")
    results, seg_samples, bbox_records = evaluate(model, test_loader, args, device)

    # Log metrics to Summary
    print("\n"  + "\nTEST RESULTS\n" )
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
        wandb.run.summary[k] = v
    wandb.log(results)

    # Section 2.6: Segmentation Samples
    if seg_samples:
        wandb.log({"test/seg_samples": [wandb.Image(np.concatenate([_denorm_image(s["image"]), _mask_to_rgb(s["gt_mask"]), _mask_to_rgb(s["pred_mask"])], axis=1), caption="Orig | GT | Pred") for s in seg_samples]})

    # Section 2.5: Bounding Box Table
    if bbox_records:
        table = wandb.Table(columns=["image", "gt_box", "pred_box", "iou", "confidence"])
        for r in bbox_records:
            vis = copy.deepcopy(_denorm_image(r["image"]))
            _draw_box(vis, r["gt_box"], (0, 255, 0), 224, 224); _draw_box(vis, r["pred_box"], (255, 0, 0), 224, 224)
            table.add_data(wandb.Image(vis), str(r["gt_box"].tolist()), str(r["pred_box"].tolist()), round(r["iou"], 4), round(1 - np.linalg.norm(r["pred_box"][:2] - r["gt_box"][:2]), 4))
        wandb.log({"test/bbox_table": table})

    # Section 2.7: In-the-wild Pipeline Showcase
    if os.path.exists("wild_images") and args.mode in ("multitask", "segmentation"):
        print("\nRunning wild image inference...")
        wild_imgs = [os.path.join("wild_images", f) for f in os.listdir("wild_images") if f.lower().endswith((".jpg", ".png"))][:3]
        trans = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
        wild_res = []
        for path in wild_imgs:
            img_pil = PILImage.open(path).convert("RGB")
            inp = trans(image=np.array(img_pil))["image"].unsqueeze(0).to(device)
            out = model(inp)
            res_pan = [_denorm_image(inp[0].cpu()), _mask_to_rgb(out["segmentation"][0].argmax(0).cpu())]
            wild_res.append(wandb.Image(np.concatenate(res_pan, axis=1), caption=f"Wild: {path}"))
        wandb.log({"test/wild_images": wild_res})

    wandb.finish()

if __name__ == "__main__":
    main()