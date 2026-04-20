import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class DubaiDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = list(sorted(os.listdir(img_dir)))
        self.masks = list(sorted(os.listdir(mask_dir)))

    def __len__(self): # Length function to know how many images are in the dataset
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        num_instances, instance_mask = cv2.connectedComponents(mask) # Label connected components to identify individual buildings
        obj_ids = np.unique(instance_mask)[1:] 
        
        masks = instance_mask == obj_ids[:, None, None] # Create a binary mask for each building instance
        
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1]) 
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            
            if xmax == xmin: xmax += 1
            if ymax == ymin: ymax += 1
                
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) # Convert bounding boxes to PyTorch tensor format
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # Calculate area of each box

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area
        }
        
        img_tensor = F.to_tensor(img)

        return img_tensor, target