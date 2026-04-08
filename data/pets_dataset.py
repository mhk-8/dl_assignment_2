"""Dataset skeleton for Oxford-IIIT Pet.
"""
import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        clip=True,
        min_visibility=0.1,
    ))


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        clip=True,
        min_visibility=0.1,
    ))

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    def __init__(self, root: str, split: str = "train", transform=None):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test'. Got: '{split}'"
            
        self.root = root
        self.split = split
        
       
        self.images_dir = os.path.join(root, "images")
        self.xmls_dir   = os.path.join(root, "annotations", "xmls")
        self.trimaps_dir= os.path.join(root, "annotations", "trimaps")
        # Parse annotations/list.txt
        # Format: <image_name> <CLASS-ID> <SPECIES> <BREED-ID>
        # CLASS-ID is 1-indexed (1-37) → convert to 0-indexed 
        ann_file = os.path.join(root, "annotations", "list.txt")
        samples = []
        with open(ann_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_name = parts[0]
                class_id = int(parts[1]) - 1    # ← parts[1] = CLASS-ID (not SPECIES)
                samples.append((img_name, class_id))
        
        # Perform Random Split (e.g., 80/20 for Train/Val)
        # Requirement: Proper and random split isolation 
        rng = np.random.default_rng(42)
        indices = np.arange(len(samples))
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)

        if split == "train":
            chosen = indices[:n_train]
        elif split == "val":
            chosen = indices[n_train:n_train + n_val]
        else:  # test
            chosen = indices[n_train + n_val:]
        self.samples = [samples[i] for i in chosen]
        
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()
        

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_xml(self, xml_path:str, img_w: int, img_h: int):
        """Parse Pascal VOC XML to get [xmin, ymin, xmax, ymax]."""
        tree   = ET.parse(xml_path)
        root   = tree.getroot()
        obj    = root.find("object")
        if obj is None:
            return [112.0, 112.0, 224.0, 224.0]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            return [112.0, 112.0, 224.0, 224.0]
        
        # Get raw pixel coords from XML
        xmin = float(bndbox.find("xmin").text) 
        ymin = float(bndbox.find("ymin").text) 
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text) 
        
        # Scale to 224×224 target size
        xmin = xmin / img_w * 224.0
        ymin = ymin / img_h * 224.0
        xmax = xmax / img_w * 224.0
        ymax = ymax / img_h * 224.0
        
        # Convert to (cx, cy, w, h) in pixel space
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        bw =  xmax - xmin
        bh =  ymax - ymin
        return [
            float(np.clip(cx, 0, 224)),
            float(np.clip(cy, 0, 224)),
            float(np.clip(bw, 0, 224)),
            float(np.clip(bh, 0, 224)),
        ]

    def __getitem__(self, idx) -> dict:
        img_name, class_id = self.samples[idx]
        
        # Load Image
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, f"{img_name}.png")
        image  = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = image.shape[:2]
        
        # Load Trimap Mask
        mask_path = os.path.join(self.trimaps_dir, f"{img_name}.png")
        mask = np.array(Image.open(mask_path))
        mask = np.where(mask == 1, 0, np.where(mask == 2, 1, 2)).astype(np.uint8) 
        
        # Load Bounding Box [xmin, ymin, xmax, ymax]
        xml_path = os.path.join(self.xmls_dir, f"{img_name}.xml")
        bbox_yolo = (self._parse_xml(xml_path, img_w, img_h)
                 if os.path.exists(xml_path)
                 else [112.0, 112.0, 224.0, 224.0])
        
        # Albumentations - needs normalized YOLO internally
        bbox_normalized = [
        bbox_yolo[0] / 224.0,
        bbox_yolo[1] / 224.0,
        bbox_yolo[2] / 224.0,
        bbox_yolo[3] / 224.0,
        ]
        augmented = self.transform(
            image=image,
            mask=mask,
            bboxes=[bbox_normalized],
            class_labels=[class_id],
        )

        image  = augmented["image"]                      
        mask   = augmented["mask"].long()                       
        bboxes = augmented["bboxes"]
        
        # After augmentation, convert albumentations normalized output → pixels
        if len(bboxes) > 0:
            cx_n, cy_n, w_n, h_n = bboxes[0]
            bbox = torch.tensor([
                cx_n * 224.0 ,
                cy_n * 224.0 ,
                w_n  * 224.0,
                h_n * 224.0
            ], dtype=torch.float32)
        else:
            bbox = torch.tensor([112.0, 112.0, 224.0, 224.0], dtype=torch.float32)
            
        return {
            "image": image,
            "label": torch.tensor(class_id, dtype=torch.long),
            "bbox": bbox,
            "mask": mask
        }