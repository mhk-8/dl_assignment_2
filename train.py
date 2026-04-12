
import argparse, os, time, torch, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data.pets_dataset import OxfordIIITPetDataset, get_train_transforms, get_val_transforms
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--task", type=str, default="localization")
    p.add_argument("--dropout_p", type=float, default=0.3) 
    p.add_argument("--freeze_encoder", type=str, default="full")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4) 
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--cls_ckpt", type=str, default=None)
    return p.parse_args()

def compute_iou_batch(pred_boxes, target_boxes, eps=1e-6):
    p_x1, p_y1 = pred_boxes[:, 0] - pred_boxes[:, 2]/2, pred_boxes[:, 1] - pred_boxes[:, 3]/2
    p_x2, p_y2 = pred_boxes[:, 0] + pred_boxes[:, 2]/2, pred_boxes[:, 1] + pred_boxes[:, 3]/2
    t_x1, t_y1 = target_boxes[:, 0] - target_boxes[:, 2]/2, target_boxes[:, 1] - target_boxes[:, 3]/2
    t_x2, t_y2 = target_boxes[:, 0] + target_boxes[:, 2]/2, target_boxes[:, 1] + target_boxes[:, 3]/2
    i_x1, i_y1 = torch.max(p_x1, t_x1), torch.max(p_y1, t_y1)
    i_x2, i_y2 = torch.min(p_x2, t_x2), torch.min(p_y2, t_y2)
    inter = torch.clamp(i_x2 - i_x1, 0) * torch.clamp(i_y2 - i_y1, 0)
    union = (pred_boxes[:, 2] * pred_boxes[:, 3]) + (target_boxes[:, 2] * target_boxes[:, 3]) - inter + eps
    return (inter + eps) / (union + eps)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(OxfordIIITPetDataset(args.data_root, "train", get_train_transforms()), args.batch_size, True, num_workers=args.num_workers)
    val_loader = DataLoader(OxfordIIITPetDataset(args.data_root, "val", get_val_transforms()), args.batch_size, False, num_workers=args.num_workers)
    
    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)
    
    payload = torch.load(args.cls_ckpt, map_location=device)
    model.encoder.load_state_dict(payload.get("state_dict", payload), strict=False)
    for p in model.encoder.parameters(): p.requires_grad = False
        
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    loc_crit = IoULoss(reduction="mean").to(device)
    dist_crit = nn.SmoothL1Loss().to(device)
    
    best_m = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss = 0
        for batch in train_loader:
            img, box = batch["image"].to(device), batch["bbox"].to(device)
            optimizer.zero_grad()
            
            # CRITICAL FIX: Scale the [0,1] model output to [0,224] for the loss calculation
            pred_box_norm = model(img)
            pred_box_pixel = pred_box_norm * 224.0 
            
            dist_loss = dist_crit(pred_box_pixel, box)
            iou_loss = loc_crit(pred_box_pixel, box)
            loss = dist_loss + (iou_loss * 5.0)
            
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
        model.eval()
        v_loss, ious = 0, []
        with torch.no_grad():
            for batch in val_loader:
                img, box = batch["image"].to(device), batch["bbox"].to(device)
                
                pred_box_norm = model(img)
                pred_box_pixel = pred_box_norm * 224.0
                
                dist_loss = dist_crit(pred_box_pixel, box)
                iou_loss = loc_crit(pred_box_pixel, box)
                loss = dist_loss + (iou_loss * 5.0)
                
                v_loss += loss.item()
                ious.extend(compute_iou_batch(pred_box_pixel, box).cpu().tolist())
                
        mean_iou = float(np.mean(ious))
        print(f"Epoch {epoch}/{args.epochs} | train/loss={t_loss/len(train_loader):.4f} | val/loss={v_loss/len(val_loader):.4f} | val/mean_iou={mean_iou:.4f}")

        if mean_iou > best_m:
            best_m = mean_iou
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "best_metric": best_m}, os.path.join(args.ckpt_dir, "localizer.pth"))

if __name__ == "__main__":
    main()