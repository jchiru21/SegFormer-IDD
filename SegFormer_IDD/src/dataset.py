import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor

IGNORE_INDEX = 255

class IDDSegmentationDataset(Dataset):
    def __init__(self, images_root, masks_root, transform=None, label_remap=None, size=(512,512)):
        self.images_root = Path(images_root)
        self.masks_root  = Path(masks_root)
        self.transform   = transform
        self.label_remap = label_remap
        self.size        = size

        self.images = []
        self.masks  = []

        for city_folder in os.listdir(self.images_root):
            city_img_dir  = self.images_root / city_folder
            city_mask_dir = self.masks_root / city_folder
            if not city_mask_dir.exists():
                continue

            for img_file in os.listdir(city_img_dir):
                if img_file.endswith("_leftImg8bit.png"):
                    mask_file = img_file.replace("_leftImg8bit.png", "_labelIds.png")
                    mask_path = city_mask_dir / mask_file
                    if not mask_path.exists():
                        continue
                    self.images.append(city_img_dir / img_file)
                    self.masks.append(mask_path)

        print(f"Dataset initialized with {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB").resize(self.size)
        mask = Image.open(self.masks[idx]).resize(self.size, resample=Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)

        # Remap labels
        if self.label_remap:
            remapped = np.ones_like(mask, dtype=np.int64) * IGNORE_INDEX
            for k, v in self.label_remap.items():
                remapped[mask == k] = v
            mask = remapped

        if self.transform:
            img = self.transform(img)

        return {"pixel_values": img, "labels": torch.tensor(mask, dtype=torch.long)}

