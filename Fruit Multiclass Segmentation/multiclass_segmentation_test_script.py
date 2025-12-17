import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

# ==========================================
# CONFIGURATION
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fruit_to_class = {
    "background": 0, "apple_Gala":1, "apple_Golden Delicious":2, "Avocado":3, "Banana":4,
    "Berry":5, "Burmese Grape":6, "Carambola":7, "Date Palm":8, "Dragon":9, "Elephant Apple":10,
    "Grape":11, "Green Coconut":12, "Guava":13, "Hog Plum":14, "Kiwi":15, "Lichi":16, "Malta":17,
    "Mango Golden Queen":18, "Mango_Alphonso":19, "Mango_Amrapali":20, "Mango_Bari":21,
    "Mango_Himsagar":22, "Olive":23, "Orange":24, "Palm":25, "Persimmon":26, "Pineapple":27,
    "Pomegranate":28, "Watermelon":29, "White Pear":30
}
class_to_fruit = {v: k for k, v in fruit_to_class.items()}

colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
extra_colors = plt.cm.Set3(np.linspace(0, 1, 11))[:, :3]
all_colors = np.vstack(([[0,0,0]], colors, extra_colors))[:31]
custom_cmap = mcolors.ListedColormap(all_colors)
def mask_to_color(mask, n_classes=31):
    cmap = custom_cmap
    color_mask = cmap(mask / (n_classes-1))[:, :, :3]
    return (color_mask * 255).astype(np.uint8)

def predict_multiclass_part_E(image_paths, model_path, device, output_dir='output_part_E'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Re-initialize model architecture
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=31
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    report_file = os.path.join(output_dir, 'predictions.txt')
    
    with open(report_file, 'w') as f:
        for image_path in image_paths:
            if not os.path.exists(image_path):
                continue
                
            img_name = os.path.basename(image_path)
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # Inference
            aug = A.Compose([A.Resize(512,512), A.Normalize(), ToTensorV2()])
            tensor = aug(image=img_rgb)["image"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(tensor)
                mask = pred.argmax(1).squeeze().cpu().numpy()
            
            # Resize mask back to original image size
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Save Colored Mask 
            mask_colored = mask_to_color(mask_resized)
            save_path = os.path.join(output_dir, f"mask_{img_name}")
            plt.imsave(save_path, mask_colored)
            
            # Analyze Classes
            unique_classes = np.unique(mask_resized)
            # Filter background (0)
            fruit_classes = [c for c in unique_classes if c != 0]
            
            if len(fruit_classes) > 0:
                # Find dominant fruit
                counts = np.bincount(mask_resized.flatten())
                # Zero out background count to ignore it for dominance
                counts[0] = 0 
                dominant_idx = np.argmax(counts)
                dominant_fruit = class_to_fruit[dominant_idx]
                detected_fruits = [class_to_fruit[c] for c in fruit_classes]
            else:
                dominant_fruit = "None"
                detected_fruits = []

            # Write to text file [cite: 45]
            f.write(f"Image: {img_name}\n")
            f.write(f"Dominant Fruit: {dominant_fruit}\n")
            f.write(f"Detected Classes: {detected_fruits}\n")
            f.write("-" * 30 + "\n")
            
    print(f"Predictions saved to {output_dir}")

# Example Usage for Test Script
test_images = ["images (1).jpg"]
predict_multiclass_part_E(test_images, "Fruit Multiclass Segmentation\\best_model_multiclass.pth", device)