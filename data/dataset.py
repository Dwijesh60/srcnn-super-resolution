import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from glob import glob

class SRCNNDataset(Dataset):
    def __init__(self, image_dir, split='train'):
        """
        split: 'train' (60 images), 'val' (20 images), 'test' (20 images)
        """
        self.split = split
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.png')))
        
        # Ensure we only take 100 images if there are more
        if len(self.image_paths) > 100:
            self.image_paths = self.image_paths[:100]
            
        if len(self.image_paths) == 100:
            if split == 'train':
                self.image_paths = self.image_paths[:60]
            elif split == 'val':
                self.image_paths = self.image_paths[60:80]
            elif split == 'test':
                self.image_paths = self.image_paths[80:]
        else:
            print(f"Warning: Expected 100 images, found {len(self.image_paths)}. Split might be uneven.")
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import random
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # 1. Center crop full images to 800x800.
        transform_crop = transforms.CenterCrop(800)
        img_cropped = transform_crop(img)
        
        # Data Augmentation: Random Flips for Training
        if self.split == 'train':
            if random.random() < 0.5:
                img_cropped = img_cropped.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img_cropped = img_cropped.transpose(Image.FLIP_TOP_BOTTOM)
                
        # 2. Downsize HR to 224x224
        img_hr = img_cropped.resize((224, 224), Image.BICUBIC)
        
        # 3. Create LR (downscale to 112x112 then upsample to 224x224)
        img_lr = img_hr.resize((112, 112), Image.BILINEAR)
        img_lr_up = img_lr.resize((224, 224), Image.BICUBIC)
        
        # 4. Normalize pixel values to [0.0, 1.0]. ToTensor() does this automatically.
        transform_tensor = transforms.ToTensor()
        hr_tensor = transform_tensor(img_hr)
        lr_up_tensor = transform_tensor(img_lr_up)
        
        # Also return the 112x112 LR tensor for baseline comparisons
        lr_tensor = transform_tensor(img_lr)
        
        return lr_up_tensor, hr_tensor, lr_tensor
