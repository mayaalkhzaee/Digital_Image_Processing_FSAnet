import cv2
import os
import numpy as np
import glob

def patch_dubai_dataset_nested(base_dir, out_img_dir, out_mask_dir, patch_size=256):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    building_color = np.array([152, 16, 60]) # MBRSC uses a dark purple hex for buildings.
    patch_count = 0
    
    tile_folders = glob.glob(os.path.join(base_dir, 'Tile *'))
    
    for tile_dir in tile_folders:
        print(f"Scanning {os.path.basename(tile_dir)}")
        image_dir = os.path.join(tile_dir, 'images')
        mask_dir = os.path.join(tile_dir, 'masks')
        
        if not os.path.exists(image_dir): continue # Skip if no images folder
        
        for img_path in glob.glob(os.path.join(image_dir, '*.jpg')):
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace('.jpg', '.png') # The masks have the exact same name but use .png 
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(mask_path): continue
            
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            
            h, w = img.shape[:2]
            
            for y in range(0, h, patch_size): # Slide a window across the image in steps of patch_size
                for x in range(0, w, patch_size):
                    img_patch = img[y:y+patch_size, x:x+patch_size] # Extract the corresponding patch from the image
                    mask_patch = mask[y:y+patch_size, x:x+patch_size]
                    
                    if img_patch.shape[:2] == (patch_size, patch_size): # Only keep full-size patches
                        # Create a binary mask where pixels matching the building color are white and everything else is black
                        binary_mask = cv2.inRange(mask_patch, building_color, building_color)
                        
                        # Only save if there's actually a good chunk of a building in the patch
                        if cv2.countNonZero(binary_mask) > 100: 
                            patch_filename = f"patch_{patch_count}.png"
                            cv2.imwrite(os.path.join(out_img_dir, patch_filename), img_patch)
                            cv2.imwrite(os.path.join(out_mask_dir, patch_filename), binary_mask)
                            patch_count += 1

    print(f"Success! Generated {patch_count} training patches.")

base_directory = r'dubai_dataset\Semantic segmentation dataset'
patch_dubai_dataset_nested(base_directory, 'data/train/images', 'data/train/masks')