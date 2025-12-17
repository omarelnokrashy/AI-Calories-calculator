import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import torch.optim as optim
import matplotlib.colors as mcolors
import time
import random
from segmentation_models_pytorch.metrics import get_stats, iou_score
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & CLASSES
# ==========================================
fruit_to_class = {
    "background": 0, "apple_Gala":1, "apple_Golden Delicious":2, "Avocado":3, "Banana":4,
    "Berry":5, "Burmese Grape":6, "Carambola":7, "Date Palm":8, "Dragon":9, "Elephant Apple":10,
    "Grape":11, "Green Coconut":12, "Guava":13, "Hog Plum":14, "Kiwi":15, "Lichi":16, "Malta":17,
    "Mango Golden Queen":18, "Mango_Alphonso":19, "Mango_Amrapali":20, "Mango_Bari":21,
    "Mango_Himsagar":22, "Olive":23, "Orange":24, "Palm":25, "Persimmon":26, "Pineapple":27,
    "Pomegranate":28, "Watermelon":29, "White Pear":30
}
class_to_fruit = {v: k for k, v in fruit_to_class.items()}

# ==========================================
# 2. DATASET (ROBUST MATCHING)
# ==========================================
class FruitSegmentationDataset(Dataset):
    def __init__(self, root_dir, fruit_to_class=None, transform=None):
        self.samples = []
        self.transform = transform

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
            
        fruit_classes = sorted(os.listdir(root_dir))
        if fruit_to_class is None:
            self.fruit_to_class = {fruit: idx+1 for idx, fruit in enumerate(fruit_classes)}
        else:
            self.fruit_to_class = fruit_to_class

        for fruit in fruit_classes:
            fruit_folder = os.path.join(root_dir, fruit)
            img_folder = os.path.join(fruit_folder, "Images")
            mask_folder = os.path.join(fruit_folder, "Mask")
            
            if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
                continue

            img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

            for img_file in img_files:
                file_id = os.path.splitext(img_file)[0]
                # Robust matching for _mask suffix or standard names
                potential_masks = [
                    file_id + "_mask.png", file_id + "_mask.jpg",
                    file_id + ".png", file_id + ".jpg"
                ]
                
                mask_path = None
                for mask_name in potential_masks:
                    full_path = os.path.join(mask_folder, mask_name)
                    if os.path.exists(full_path):
                        mask_path = full_path
                        break
                
                if mask_path:
                    self.samples.append({
                        "img": os.path.join(img_folder, img_file),
                        "mask": mask_path,
                        "class_id": self.fruit_to_class[fruit]
                    })
                else:
                    print(f"Skipping {img_file}: Could not find {potential_masks[0]} or others")
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = cv2.imread(sample["img"])
        if img is None: raise FileNotFoundError(f"Failed to load image: {sample['img']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(sample["mask"], 0)
        if mask is None: raise FileNotFoundError(f"Failed to load mask: {sample['mask']}")

        # Resize to 512x512 for high resolution
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 127).astype("uint8") * sample["class_id"]

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = torch.as_tensor(augmented["mask"], dtype=torch.long)
        else:
            img = transforms.ToTensor()(img)
            mask = torch.tensor(mask, dtype=torch.long)

        return img, mask

# ==========================================
# 3. SETUP & TRAINING
# ==========================================
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

root_dir_train = "Project Data/Fruit/train"
root_dir_val = "Project Data/Fruit/Validation"

train_dataset = FruitSegmentationDataset(root_dir=root_dir_train, transform=transform)
validation_dataset = FruitSegmentationDataset(root_dir=root_dir_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(validation_dataset)}")

# Class distribution
train_classes = defaultdict(int)
for sample in train_dataset.samples:
    train_classes[sample['class_id']] += 1
print("Train class distribution:", dict(train_classes))

# Visualize samples

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor.cpu() * std + mean

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i in range(3):
    img, mask = train_dataset[i]
    img = unnormalize(img).permute(1,2,0).numpy()
    img = np.clip(img, 0, 1)
    axes[0, i].imshow(img)
    axes[0, i].set_title("Image")
    axes[1, i].imshow(mask.numpy(), cmap='tab20')
    axes[1, i].set_title("Mask")
plt.show()
plt.savefig("sample_visualizations.png")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = 31

# UPGRADE: Using EfficientNet-B4 for better feature extraction
model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=n_classes
).to(device)

dice_loss = smp.losses.DiceLoss(mode='multiclass')
ce_loss = nn.CrossEntropyLoss()

def criterion(pred, target):
    return 0.5 * dice_loss(pred, target) + 0.5 * ce_loss(pred, target)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# ==========================================
# 4. TRAINING LOOP WITH "SAVE BEST"
# ==========================================
num_epochs = 30
train_losses = []
val_losses = []
val_ious = []
max_score = 0 # Track best IoU

print("Starting Training...")
start_time = time.time()
c = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    total_tp = torch.zeros(n_classes, device=device)
    total_fp = torch.zeros(n_classes, device=device)
    total_fn = torch.zeros(n_classes, device=device)
    total_tn = torch.zeros(n_classes, device=device)
    
    with torch.no_grad():
        for imgs, masks in tqdm(validation_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)
            
            pred_masks = outputs.argmax(dim=1)
            tp, fp, fn, tn = get_stats(pred_masks, masks, mode='multiclass', num_classes=n_classes)
            
            total_tp += tp.to(device).sum(dim=0)
            total_fp += fp.to(device).sum(dim=0)
            total_fn += fn.to(device).sum(dim=0)
            total_tn += tn.to(device).sum(dim=0)

    val_loss /= len(validation_loader.dataset)
    val_losses.append(val_loss)

    miou = iou_score(total_tp, total_fp, total_fn, total_tn, reduction='macro')
    val_iou = miou.item()
    val_ious.append(val_iou)
    
    scheduler.step(val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
    # SAVE BEST MODEL LOGIC
    if val_iou > max_score:
        max_score = val_iou
        torch.save(model.state_dict(), "best_model_multiclass.pth")
        print(f"NEW BEST IoU: {val_iou:.4f} (Model Saved)")
        c = 0
    else:
        c += 1

    if c >= 5:
        print("Early stopping triggered.")
        break

print(f"Total training time: {time.time() - start_time:.2f} seconds")
print(f"Best Validation IoU: {max_score:.4f}")

# Plotting
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
plt.savefig("loss_curve.png")

# ==========================================
# 5. VISUALIZATION & INFERENCE (Part E)
# ==========================================

# Fix Colormap (Align dimensions)
colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
extra_colors = plt.cm.Set3(np.linspace(0, 1, 11))[:, :3]
all_colors = np.vstack(([[0,0,0]], colors, extra_colors))[:31]
custom_cmap = mcolors.ListedColormap(all_colors)

def mask_to_color(mask, n_classes=31):
    cmap = custom_cmap
    color_mask = cmap(mask / (n_classes-1))[:, :, :3]
    return (color_mask * 255).astype(np.uint8)

def visualize_prediction(model, dataset, idx, device):
    model.eval()
    sample = dataset.samples[idx]
    
    image = cv2.imread(sample['img'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(sample['mask'], 0)
    
    # Preprocess
    aug = A.Compose([A.Resize(512,512), A.Normalize(), ToTensorV2()])
    tensor = aug(image=image)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

    # Plot
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.title("Original"); plt.imshow(image); plt.axis('off')
    plt.subplot(1,3,2); plt.title(f"Ground Truth ({class_to_fruit[sample['class_id']]})"); plt.imshow(mask_to_color(gt_mask)); plt.axis('off')
    plt.subplot(1,3,3); plt.title("Prediction"); plt.imshow(mask_to_color(pred_mask)); plt.axis('off')
    plt.show()
    plt.savefig(f"visualization_{idx}.png")

# VISUALIZE RANDOM SAMPLES (Not just the first 5)
print("Visualizing Random Validation Samples...")
model.load_state_dict(torch.load("best_model_multiclass.pth")) # Load best weights
indices = random.sample(range(len(validation_dataset)), 5)
for idx in indices:
    visualize_prediction(model, validation_dataset, idx, device)
