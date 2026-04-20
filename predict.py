import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from model import get_baseline_model 
import numpy as np

def generate_report_images():
    device = torch.device('cpu') 
    model = get_baseline_model(num_classes=2)
    
    model.load_state_dict(torch.load('baseline_maskrcnn.pth', map_location=device))
    model.eval() 

    img_path = 'data/train/images/patch_2.png' 
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert image to PyTorch tensor format
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    # Only keep predictions that the model is more than 50% confident about
    masks = prediction['masks'][prediction['scores'] > 0.5].squeeze(1).cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    
    for mask in masks:
        # Mask out the 0s so they are transparent
        transparent_mask = np.ma.masked_where(mask <= 0.5, mask)
        
        # Overlay only the building footprint
        plt.imshow(transparent_mask, cmap='Reds', alpha=0.5)
    
    plt.title(f"Baseline Found {len(masks)} Buildings")
    plt.axis('off')
    
    plt.savefig('report_baseline_result_2.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    generate_report_images()