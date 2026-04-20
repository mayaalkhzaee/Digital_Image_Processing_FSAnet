import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from newmodel import get_fsanet_model 

def generate_fsanet_images():
    device = torch.device('cpu') 
    
    model = get_fsanet_model(num_classes=2)
    
    model.load_state_dict(torch.load('fsanet_maskrcnn.pth', map_location=device))
    model.eval()

    img_path = 'data/train/images/patch_2.png' 
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    masks = prediction['masks'][prediction['scores'] > 0.5].squeeze(1).cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    
    for mask in masks:
        transparent_mask = np.ma.masked_where(mask <= 0.5, mask)
        plt.imshow(transparent_mask, cmap='Greens', alpha=0.5) 
    
    plt.title(f"FsaNet Found {len(masks)} Buildings")
    plt.axis('off')
    
    plt.savefig('report_fsanet_result_2.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    generate_fsanet_images()