
"""Inference and evaluation
"""
import argparse
import os
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import wandb
from PIL import Image as PILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset   import OxfordIIITPetDataset, get_val_transforms
from multitask           import MultiTaskPerceptionModel
from models.classification import VGG11Classifier
from models.localization   import VGG11Localizer
from models.segmentation   import VGG11UNet


def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 Inference")
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--cls_ckpt",    type=str, default="checkpoints/classifier.pth")
    p.add_argument("--loc_ckpt",    type=str, default="checkpoints/localizer.pth")
    p.add_argument("--seg_ckpt",    type=str, default="checkpoints/unet.pth")
    p.add_argument("--num_classes", type=int, default=37)
    p.add_argument("--seg_classes", type=int, default=3)
    p.add_argument("--dropout_p",   type=float, default=0.5)
    p.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--mode", type=str, default="multitask",
                   choices=["multitask","classification","localization","segmentation"])
    p.add_argument("--use_bn", type=lambda x: x.lower() != "false", default=True)
    return p.parse_args()


def _denorm_image(t):
    img = (t.permute(1,2,0).numpy() *
           np.array([0.229,0.224,0.225]) +
           np.array([0.485,0.456,0.406])) * 255
    return np.clip(img, 0, 255).astype(np.uint8)

_SEG_PALETTE = np.array([[255,255,255],[0,0,0],[128,128,128]], dtype=np.uint8)

def _mask_to_rgb(mask):
    m = np.clip(mask.numpy().astype(np.int64), 0, 2)
    return _SEG_PALETTE[m]

def _draw_box(img, box, color, thickness=2):
    cx, cy, bw, bh = box
    x1, y1 = int(cx-bw/2), int(cy-bh/2)
    x2, y2 = int(cx+bw/2), int(cy+bh/2)
    H, W = img.shape[:2]
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(W-1,x2), min(H-1,y2)
    img[y1:y1+thickness,x1:x2]=color
    img[y2:y2+thickness,x1:x2]=color
    img[y1:y2,x1:x1+thickness]=color
    img[y1:y2,x2:x2+thickness]=color


def compute_iou_per_sample(pred, tgt, eps=1e-6):
    px1=pred[:,0]-pred[:,2]/2; py1=pred[:,1]-pred[:,3]/2
    px2=pred[:,0]+pred[:,2]/2; py2=pred[:,1]+pred[:,3]/2
    tx1=tgt[:,0]-tgt[:,2]/2;   ty1=tgt[:,1]-tgt[:,3]/2
    tx2=tgt[:,0]+tgt[:,2]/2;   ty2=tgt[:,1]+tgt[:,3]/2
    ix1=torch.max(px1,tx1); iy1=torch.max(py1,ty1)
    ix2=torch.min(px2,tx2); iy2=torch.min(py2,ty2)
    inter=torch.clamp(ix2-ix1,0)*torch.clamp(iy2-iy1,0)
    union=pred[:,2]*pred[:,3]+tgt[:,2]*tgt[:,3]-inter+eps
    return (inter+eps)/(union+eps)

def compute_map(all_iou):
    thresholds = np.arange(0.5, 1.0, 0.05)
    iou_arr = np.array(all_iou)
    aps = [(iou_arr >= thr).sum() / len(iou_arr)
           if len(iou_arr) > 0 else 0.0 for thr in thresholds]
    return float(np.mean(aps))


def build_eval_model(args, device):
    print(f"\nBuilding evaluation model: {args.mode}")
    if args.mode == "multitask":
        model = MultiTaskPerceptionModel(
            num_breeds=args.num_classes,
            seg_classes=args.seg_classes,
            classifier_path=args.cls_ckpt,
            localizer_path =args.loc_ckpt,
            unet_path      =args.seg_ckpt,
        )
        for ckpt, attr in [(args.cls_ckpt,"cls"),(args.loc_ckpt,"loc"),(args.seg_ckpt,"seg")]:
            if not os.path.exists(ckpt): continue
            sd = torch.load(ckpt, map_location=device).get("state_dict", {})
            if attr == "cls":
                model.encoder.load_state_dict(
                    {k.replace("encoder.",""): v for k,v in sd.items() if k.startswith("encoder.")}, strict=False)
                model.cls_head.load_state_dict(
                    {k.replace("classifier.",""): v for k,v in sd.items() if k.startswith("classifier.")}, strict=False)
            elif attr == "loc":
                model.bbox_head.load_state_dict(
                    {k.replace("regression_head.",""): v for k,v in sd.items() if "regression_head" in k}, strict=False)
            else:
                model.load_state_dict(
                    {k: v for k,v in sd.items() if not k.startswith("encoder.")}, strict=False)
            print(f"    Loaded: {ckpt}")
    else:
        if args.mode == "classification":
            model = VGG11Classifier(num_classes=args.num_classes, dropout_p=args.dropout_p)
        elif args.mode == "localization":
            model = VGG11Localizer()
        elif args.mode == "segmentation":
            model = VGG11UNet(num_classes=args.seg_classes)
        ckpt_path = {"classification":args.cls_ckpt,
                     "localization":args.loc_ckpt,
                     "segmentation":args.seg_ckpt}[args.mode]
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device).get("state_dict", {}), strict=False)
        print(f"    Loaded: {ckpt_path}")
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, loader, args, device):
    all_lbl, all_prd = [], []
    all_iou          = []
    all_dice, all_acc= [], []
    seg_samples, bbox_records = [], []

    for batch in loader:
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)
        box = batch["bbox"].to(device)
        msk = batch["mask"].to(device)
        out = model(img)

        c_out = out["classification"] if isinstance(out,dict) else (out if args.mode=="classification" else None)
        l_out = out["localization"]   if isinstance(out,dict) else (out if args.mode=="localization"   else None)
        s_out = out["segmentation"]   if isinstance(out,dict) else (out if args.mode=="segmentation"   else None)

        if c_out is not None:
            all_prd.extend(c_out.argmax(1).cpu().tolist())
            all_lbl.extend(lbl.cpu().tolist())

        if l_out is not None:
            ious = compute_iou_per_sample(l_out, box)
            all_iou.extend(ious.cpu().tolist())
            if len(bbox_records) < 15:
                for i in range(len(img)):
                    if len(bbox_records) >= 15: break
                    bbox_records.append({
                        "image":    img[i].cpu(),
                        "gt_box":   box[i].cpu().numpy(),
                        "pred_box": l_out[i].cpu().numpy(),
                        "iou":      float(ious[i].cpu()),
                    })

        if s_out is not None:
            # ── FIX: move to CPU before any numpy/comparison ops ──
            pred_m  = s_out.argmax(1).cpu()
            msk_cpu = msk.cpu()

            dice_scores = []
            for c in range(args.seg_classes):
                p = (pred_m == c).float()
                t = (msk_cpu == c).float()
                dice_scores.append(
                    (2*(p*t).sum().item()) /
                    (p.sum().item() + t.sum().item() + 1e-6)
                )
            all_dice.append(float(np.mean(dice_scores)))
            all_acc.append(float((pred_m == msk_cpu).float().mean().item()))

            if len(seg_samples) < 5:
                seg_samples.append({
                    "image":     img[0].cpu(),
                    "gt_mask":   msk_cpu[0],
                    "pred_mask": pred_m[0],
                })

    res = {}
    if all_lbl: res["test/macro_f1"] = f1_score(all_lbl,all_prd,average="macro",zero_division=0)
    if all_iou:
        res["test/mean_iou"] = float(np.mean(all_iou))
        res["test/mAP"]      = compute_map(all_iou)
    if all_dice:
        res["test/dice_score"]     = float(np.mean(all_dice))
        res["test/pixel_accuracy"] = float(np.mean(all_acc))

    return res, seg_samples, bbox_records


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=args.wandb_project, name="eval_multitask",
               config=vars(args), job_type="eval")

    test_loader = DataLoader(
        OxfordIIITPetDataset(args.data_root,"test",get_val_transforms()),
        args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_eval_model(args, device)

    print("\nRunning test evaluation...")
    results, seg_samples, bbox_records = evaluate(model, test_loader, args, device)

    print("\nTEST RESULTS")
    print("="*40)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
        wandb.run.summary[k] = v
    wandb.log(results)

    if seg_samples:
        wb_imgs = []
        for s in seg_samples:
            row = np.concatenate([
                _denorm_image(s["image"]),
                _mask_to_rgb(s["gt_mask"]),
                _mask_to_rgb(s["pred_mask"]),
            ], axis=1)
            wb_imgs.append(wandb.Image(row, caption="Orig | GT | Pred"))
        wandb.log({"test/seg_samples": wb_imgs})

    if bbox_records:
        table = wandb.Table(columns=["image","gt_box","pred_box","iou","confidence"])
        for r in bbox_records:
            vis = copy.deepcopy(_denorm_image(r["image"]))
            _draw_box(vis, r["gt_box"],   (0,255,0))
            _draw_box(vis, r["pred_box"], (255,0,0))
            conf = float(max(0.0, min(1.0,
                1 - np.linalg.norm(r["pred_box"][:2]-r["gt_box"][:2])/224)))
            table.add_data(wandb.Image(vis),
                           str(r["gt_box"].tolist()),
                           str(r["pred_box"].tolist()),
                           round(r["iou"],4), round(conf,4))
        wandb.log({"test/bbox_table": table})

    wild_dir = "wild_images"
    if os.path.exists(wild_dir) and args.mode in ("multitask","segmentation"):
        print("\nRunning wild image inference...")
        wild_paths = [os.path.join(wild_dir,f)
                      for f in sorted(os.listdir(wild_dir))
                      if f.lower().endswith((".jpg",".jpeg",".png"))][:3]
        trans = A.Compose([A.Resize(224,224),
                           A.Normalize(mean=[0.485,0.456,0.406],
                                       std=[0.229,0.224,0.225]),
                           ToTensorV2()])
        wild_imgs = []
        model.eval()
        with torch.no_grad():
            for path in wild_paths:
                img_np = np.array(PILImage.open(path).convert("RGB"))
                inp    = trans(image=img_np)["image"].unsqueeze(0).to(device)
                out    = model(inp)
                orig   = np.array(PILImage.open(path).convert("RGB").resize((224,224)))
                panels = [orig]
                if isinstance(out,dict) and "localization" in out:
                    vis = orig.copy()
                    _draw_box(vis, out["localization"][0].cpu().numpy(), (255,0,0))
                    panels.append(vis)
                if isinstance(out,dict) and "segmentation" in out:
                    panels.append(_mask_to_rgb(out["segmentation"][0].argmax(0).cpu()))
                cls_id = int(out["classification"].argmax(1).item()) if isinstance(out,dict) else -1
                wild_imgs.append(wandb.Image(
                    np.concatenate(panels,axis=1),
                    caption=f"{os.path.basename(path)} | class={cls_id}"))
        if wild_imgs:
            wandb.log({"test/wild_images": wild_imgs})

    wandb.finish()
    print("\nInference complete.")

if __name__ == "__main__":
    main()