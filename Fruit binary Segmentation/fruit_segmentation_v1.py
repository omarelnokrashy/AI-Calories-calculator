from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import random
import segmentation_models_pytorch as smp

# ==================== REPRODUCIBILITY ====================
def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ==================== DATASET CLASS (FIXED) ====================
class FruitSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to 'Data/Fruit/Train' or 'Data/Fruit/Validation'
            transform: Albumentations transform
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Iterate through all fruit folders
        for fruit_name in os.listdir(root_dir):
            fruit_path = os.path.join(root_dir, fruit_name)
            if not os.path.isdir(fruit_path):
                continue
                
            images_dir = os.path.join(fruit_path, 'Images')
            masks_dir = os.path.join(fruit_path, 'Masks')  # Fixed: was 'mask'
            
            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                continue
            
            #Map by filename stem
            image_files = {Path(f).stem: os.path.join(images_dir, f) 
                          for f in os.listdir(images_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))}
            
            mask_files = {Path(f).stem: os.path.join(masks_dir, f)
                         for f in os.listdir(masks_dir)
                         if f.endswith(('.png', '.jpg', '.jpeg'))}
            
            # Match images to masks by stem
            for stem in image_files:
                if stem in mask_files:
                    self.samples.append({
                        'image': image_files[stem],
                        'mask': mask_files[stem],
                        'fruit': fruit_name
                    })
        
        print(f"Loaded {len(self.samples)} image-mask pairs from {root_dir}")
        
        # Calculate class weights for imbalanced data
        self.calculate_class_weights()
    
    def calculate_class_weights(self, sample_size=100):
        """Calculate foreground/background ratio for weighted loss"""
        if len(self.samples) == 0:
            self.pos_weight = 1.0
            return
        
        total_fg = 0
        total_bg = 0
        
        sample_indices = random.sample(range(len(self.samples)), 
                                      min(sample_size, len(self.samples)))
        
        for idx in sample_indices:
            mask = cv2.imread(self.samples[idx]['mask'], cv2.IMREAD_GRAYSCALE)
            fg_pixels = np.sum(mask > 127)
            bg_pixels = np.sum(mask <= 127)
            total_fg += fg_pixels
            total_bg += bg_pixels
        
        # pos_weight = background / foreground (to weight minority class more)
        self.pos_weight = total_bg / (total_fg + 1e-6)
        print(f"Class imbalance ratio (bg/fg): {self.pos_weight:.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image and mask
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        # Ensure binary mask (0 and 1)
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Add channel dimension to mask
        mask = mask.unsqueeze(0)
        
        return image, mask


# ==================== MODERN UNET WITH PRETRAINED ENCODER ====================
class ModernUNet(nn.Module):
    """
    UNet with pretrained encoder, GroupNorm, and Dropout
    Uses segmentation_models_pytorch for state-of-the-art architecture
    """
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', dropout=0.3):
        super().__init__()
        
        # Modern UNet with pretrained backbone
        self.model = smp.Unet(
            encoder_name=encoder_name,        # resnet34, resnet50, efficientnet-b4, etc.
            encoder_weights=encoder_weights,  # 'imagenet' for pretrained
            in_channels=3,
            classes=1,
            activation=None,  # We'll apply sigmoid manually
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return torch.sigmoid(x)


# Alternative: Custom UNet with GroupNorm (if you can't use smp)
class CustomUNetGroupNorm(nn.Module):
    """Custom UNet with GroupNorm instead of BatchNorm + Dropout"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], dropout=0.3):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout)
        
        # Encoder
        for feature in features:
            self.downs.append(self._double_conv_gn(in_channels, feature, dropout))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._double_conv_gn(features[-1], features[-1]*2, dropout)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self._double_conv_gn(feature*2, feature, dropout))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _double_conv_gn(self, in_channels, out_channels, dropout):
        """Conv block with GroupNorm and Dropout"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),  # GroupNorm
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),  # Dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
        
        return torch.sigmoid(self.final_conv(x))


# ==================== LOSS FUNCTIONS ====================
class DiceLoss(nn.Module):
    """Dice Loss with smoothing"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Weighted BCE + Dice Loss with class imbalance handling"""
    def __init__(self, pos_weight=1.0, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        # For BCE, we need logits (before sigmoid)
        # But our model outputs sigmoid, so we need to invert
        # Better: modify model to output logits or use BCELoss instead
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(torch.sigmoid(pred), target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ==================== EARLY STOPPING ====================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


# ==================== METRICS ====================
def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate Dice Coefficient"""
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate Intersection over Union"""
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)


# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    loop = tqdm(loader, desc="Training")
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks).item()
        total_iou += iou_score(outputs, masks).item()
        
        loop.set_postfix(loss=loss.item(), dice=total_dice/(loop.n+1))
    
    return (total_loss / len(loader), 
            total_dice / len(loader), 
            total_iou / len(loader))


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks).item()
            total_iou += iou_score(outputs, masks).item()
    
    return (total_loss / len(loader), 
            total_dice / len(loader), 
            total_iou / len(loader))


# ==================== MAIN TRAINING SCRIPT ====================
def main():
    # Hyperparameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    IMAGE_SIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ENCODER = 'resnet34'  # Options: resnet34, resnet50, efficientnet-b4
    DROPOUT = 0.3
    PATIENCE = 15
    
    print(f"Device: {DEVICE}")
    print(f"Encoder: {ENCODER}")
    
    # Data Augmentation (Heavy for small datasets)
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Datasets
    train_dataset = FruitSegmentationDataset('Data/Fruit/Train', transform=train_transform)
    val_dataset = FruitSegmentationDataset('Data/Fruit/Validation', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True) #2
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True) #2
    
    # Model with pretrained encoder
    model = ModernUNet(
        encoder_name=ENCODER,
        encoder_weights='imagenet',
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Loss with class imbalance handling
    criterion = CombinedLoss(
        pos_weight=train_dataset.pos_weight,
        bce_weight=0.5,
        dice_weight=0.5
    )
    criterion.bce.pos_weight = criterion.bce.pos_weight.to(DEVICE)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Mixed precision training (faster on modern GPUs)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    
    # Training Loop
    best_dice = 0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scaler
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, DEVICE
        )
        
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        print(f"\nTrain - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
            }, 'best_fruit_segmentation.pth')
            print(f"Model saved! Best Dice: {best_dice:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\n Training completed! Best Validation Dice: {best_dice:.4f}")
    
    # Save training history
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=4)


if __name__ == '__main__':
    main()
