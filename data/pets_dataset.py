"""Dataset skeleton for Oxford-IIIT Pet."""

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
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

class OxfordIIITPetDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.images_dir = os.path.join(data_dir, "images")
        self.trimaps_dir = os.path.join(data_dir, "annotations", "trimaps")
        self.xmls_dir = os.path.join(data_dir, "annotations", "xmls")
        
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = get_train_transforms() if split == "train" else get_val_transforms()
            
        self.filenames = []
        self.classes = [None] * 37 
        self.class_to_idx = {}

        list_txt_path = os.path.join(data_dir, "annotations", "list.txt")
        with open(list_txt_path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    if filename.startswith(".") or filename.startswith("mat"): continue
                    
                    class_id = int(parts[1]) - 1
                    breed_name = "_".join(filename.split("_")[:-1])
                    self.classes[class_id] = breed_name
                    self.class_to_idx[breed_name] = class_id
                    self.filenames.append(filename)

    def __len__(self):
        return len(self.filenames)

    def _parse_xml(self, xml_path: str, img_w: int, img_h: int):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is None: return [0.0, 0.0, float(img_w), float(img_h)]
        bndbox = obj.find("bndbox")
        if bndbox is None: return [0.0, 0.0, float(img_w), float(img_h)]
        
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        xmax = min(float(img_w), xmax)
        ymax = min(float(img_h), ymax)
        
        if xmax <= xmin: xmax = xmin + 1.0
        if ymax <= ymin: ymax = ymin + 1.0
            
        return [xmin, ymin, xmax, ymax] 

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.images_dir, f"{filename}.jpg")
        
        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]

        mask_path = os.path.join(self.trimaps_dir, f"{filename}.png")
        mask = np.array(Image.open(mask_path))
        mask = np.where(mask == 1, 0, np.where(mask == 2, 1, 2)).astype(np.uint8)

        xml_path = os.path.join(self.xmls_dir, f"{filename}.xml")
        bbox_pascal = self._parse_xml(xml_path, img_w, img_h) if os.path.exists(xml_path) else [0.0, 0.0, float(img_w), float(img_h)]
        
        breed_name = "_".join(filename.split("_")[:-1])
        label_idx = self.class_to_idx[breed_name]

        augmented = self.transforms(image=img, mask=mask, bboxes=[bbox_pascal], class_labels=[label_idx])
        image = augmented["image"]                      
        mask = augmented["mask"].long()                       
        bboxes = augmented["bboxes"]
        
        # Albumentations outputs pascal_voc here because we told it to.
        # Now we manually convert it to [cx, cy, w, h] pixel coordinates.
        if len(bboxes) > 0:
            xmin, ymin, xmax, ymax = bboxes[0]
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        else:
            bbox = torch.tensor([112.0, 112.0, 224.0, 224.0], dtype=torch.float32)

        return {"image": image, "label": torch.tensor(label_idx, dtype=torch.long), "bbox": bbox, "mask": mask}