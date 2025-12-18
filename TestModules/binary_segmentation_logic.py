import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp

class ModernUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights=None, dropout=0.3):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.dropout = nn.Dropout2d(p=dropout)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return torch.sigmoid(x)

class FruitBinarySegmenter:
    def __init__(self, model_path, device=None, image_size=256, threshold=0.5):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.threshold = threshold
        
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.model = ModernUNet(encoder_name='resnet34', encoder_weights=None)
        self.model = self._load_checkpoint(self.model, model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, model, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_state_dict[key] = value
            elif not key.startswith("model.") and hasattr(model, 'model'):
                new_state_dict["model." + key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        return model

    def predict(self, image_path):
        """
        Processes an image and returns a 3-panel visualization: Original | Mask | Overlay
        """
        original_img_bgr = cv2.imread(image_path)
        if original_img_bgr is None: return None
        
        original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=original_img_rgb)
        img_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            pred_mask = (output > self.threshold).float().cpu().numpy()[0, 0, :, :]
        
        viz_img = cv2.resize(original_img_bgr, (self.image_size, self.image_size))
        
        mask_colored = np.zeros_like(viz_img)
        mask_colored[:, :, 2] = (pred_mask * 255).astype(np.uint8) 
        overlay = cv2.addWeighted(viz_img, 0.7, mask_colored, 0.3, 0)
        
        mask_3ch = cv2.cvtColor((pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        return np.hstack((viz_img, mask_3ch, overlay))
        