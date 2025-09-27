import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, images_path:pd.DataFrame,class_ids:dict, transform=None):
        self.images_path = images_path
        self.image_files = images_path['Path'].to_list()
        self.labels = images_path['Label']
        self.transform = transform
        self.class_ids = class_ids

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file).convert("RGB")
        label_str = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        label = []
        for i in range(5):
            if i < len(label_str):
                ch = label_str[i]
                label.append(self.class_ids.get(ch,0))
            else:
                label.append(0)
        
            
        return image, torch.tensor(label, dtype=torch.long)