import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torchvision.transforms.functional as F
from model import get_baseline_model
from newmodel import get_fsanet_model

def apply_mask(image, mask, color, alpha=0.5):
    out_image = image.copy()
    idx = np.nonzero(mask > 0.5)
    
    out_image[idx[0], idx[1], :] = (
        (1 - alpha) * out_image[idx[0], idx[1], :] + 
        alpha * np.array(color)
    )
    return out_image

def generate_batch_comparisons(num_images_to_show=3):
    device = torch.device('cpu')
    print(f"Running comparison script on {device}...")


    img_dir = 'data/train/images/'
    mask_dir = 'data/train/masks/'
    baseline_weights = 'baseline_maskrcnn.pth'
    fsanet_weights = 'fsanet_maskrcnn.pth'

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"Error: Could not find data directories. Check '{img_dir}' and '{mask_dir}'.")
        return

    print("Loading Baseline model...")
    baseline_model = get_baseline_model(num_classes=2)
    baseline_model.load_state_dict(torch.load(baseline_weights, map_location=device))
    baseline_model.eval()

    print("Loading FsaNet model...")
    fsanet_model = get_fsanet_model(num_classes=2)
    fsanet_model.load_state_dict(torch.load(fsanet_weights, map_location=device))
    fsanet_model.eval()

    all_images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    if len(all_images) < num_images_to_show:
        print(f"Warning: Only found {len(all_images)} images. Showing all.")
        num_images_to_show = len(all_images)
        
    selected_indices = random.sample(range(len(all_images)), num_images_to_show)
    print(f"Randomly selected {num_images_to_show} image patches: {[all_images[i] for i in selected_indices]}")

    for i, idx in enumerate(selected_indices):
        img_name = all_images[idx]
        img_path = os.path.join(img_dir, img_name)
        
        mask_path = os.path.join(mask_dir, img_name)

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

        print(f"[{i+1}/{num_images_to_show}] Analyzing {img_name}...")
        with torch.no_grad():
            baseline_pred = baseline_model(img_tensor)[0]
            fsanet_pred = fsanet_model(img_tensor)[0]

        baseline_masks = baseline_pred['masks'][baseline_pred['scores'] > 0.5].squeeze(1).cpu().numpy()
        fsanet_masks = fsanet_pred['masks'][fsanet_pred['scores'] > 0.5].squeeze(1).cpu().numpy()

        baseline_display = img_rgb.copy()
        for mask in baseline_masks:
            baseline_display = apply_mask(baseline_display, mask, color=(0, 0, 255))

        fsanet_display = img_rgb.copy()
        for mask in fsanet_masks:
            fsanet_display = apply_mask(fsanet_display, mask, color=(0, 255, 0))

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"Comparison: {img_name}", fontsize=16, fontweight='bold')

        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title("1. Original Image")
        axs[0, 0].axis('off')

        axs[0, 1].imshow(ground_truth_mask, cmap='gray')
        axs[0, 1].set_title("2. Ground Truth (Ideal Mask)")
        axs[0, 1].axis('off')

        axs[1, 0].imshow(baseline_display)
        axs[1, 0].set_title(f"3. Baseline Output (Found {len(baseline_masks)} Buildings)")
        axs[1, 0].axis('off')

        axs[1, 1].imshow(fsanet_display)
        axs[1, 1].set_title(f"4. FsaNet Output (Found {len(fsanet_masks)} Buildings)")
        axs[1, 1].axis('off')

        save_name = f'comparison_{i}.png'
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved summary figure to '{save_name}'!")

    print("\nBatch comparison complete. All random images processed.")

if __name__ == '__main__':
    generate_batch_comparisons(num_images_to_show=3)