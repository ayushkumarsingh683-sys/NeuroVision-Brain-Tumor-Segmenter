import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Match mask filename
        base_name = img_name.replace('.png', '').replace('.tif', '').replace('.jpg', '')
        mask_name = None
        for f in os.listdir(self.mask_dir):
            if base_name in f:
                mask_name = f
                break
                
        mask_path = os.path.join(self.mask_dir, mask_name) if mask_name else ""

        # Using PIL to safely load images and masks
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # Convert to PyTorch tensors (HWC to CHW)
        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask