import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import random
from tqdm import tqdm

class SwinProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.swin_t(weights=None)
        backbone.head = nn.Identity()
        self.encoder = backbone

    def forward(self, x):
        features = self.encoder(x)
        return F.normalize(features, dim=1)

class SwinProtoNetClassifier:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = SwinProtoNet()
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.prototypes = None
        self.class_names = []

    def _load_weights(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading weights: {e}")


    def save_prototypes(self, save_path):
        """Saves the computed prototypes and class names to a file."""
        if self.prototypes is None:
            print("[WARN] No prototypes to save. Run precompute_prototypes first.")
            return
        
        data = {
            'prototypes': self.prototypes.cpu(), # Move to CPU for safe storage
            'class_names': self.class_names
        }
        torch.save(data, save_path)
        print(f"[INFO] Prototypes saved to {save_path}")


    def load_prototypes(self, load_path):
        """Loads prototypes from a file, skipping computation."""
        if not os.path.exists(load_path):
            return False
        
        try:
            data = torch.load(load_path, map_location=self.device)
            self.prototypes = data['prototypes'].to(self.device)
            self.class_names = data['class_names']
            print(f"[INFO] Loaded cached prototypes for {len(self.class_names)} classes.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load prototypes: {e}")
            return False

    def precompute_prototypes(self, support_dir, shots=3):
        if not os.path.exists(support_dir):
            return

        print(f"[INFO] Building {shots}-shot prototypes from {support_dir}...")
        class_dirs = [d for d in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, d))]
        class_dirs.sort()
        
        prototypes_list = []
        self.class_names = []

        with torch.no_grad():
            for cls_name in tqdm(class_dirs, desc="Processing Classes"):
                cls_path = os.path.join(support_dir, cls_name)
                all_images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                selected_files = random.sample(all_images, min(len(all_images), shots))
                
                images = []
                for img_file in selected_files:
                    try:
                        img = Image.open(os.path.join(cls_path, img_file)).convert('RGB')
                        images.append(self.transform(img))
                    except: continue
                
                if len(images) > 0:
                    img_batch = torch.stack(images).to(self.device)
                    embeddings = self.model(img_batch)
                    proto = F.normalize(embeddings.mean(dim=0), dim=0)
                    prototypes_list.append(proto)
                    self.class_names.append(cls_name)

        if prototypes_list:
            self.prototypes = torch.stack(prototypes_list)
            print(f"[SUCCESS] Computed prototypes for {len(self.class_names)} classes.")

    def predict(self, image_path):
        if self.prototypes is None: return "Error"
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                query_emb = self.model(img_t)
                similarities = torch.mm(query_emb, self.prototypes.t()) * 20.0
                probs = F.softmax(similarities, dim=1)
                conf, idx = torch.max(probs, dim=1)
                return self.class_names[idx.item()]
        except: return "Error"