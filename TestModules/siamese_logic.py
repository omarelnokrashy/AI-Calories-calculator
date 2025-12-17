import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

class SwinProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.swin_t(weights=None)
        backbone.head = nn.Identity()
        self.encoder = backbone

    def forward(self, x):
        features = self.encoder(x)
        return F.normalize(features, dim=1)

class SwinProtoNetSiamese:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Setup Transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 2. Load the Model
        self.model = SwinProtoNet()
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different saving formats
        state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_most_similar(self, folder_path):
        """
        Takes a folder path, finds the 'anchor' image, and returns the filename 
        of the most similar reference image and its similarity score.
        """
        all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not all_files:
            return None, 0.0

        anchor_file = next((f for f in all_files if 'anchor' in f.lower()), None)
        if not anchor_file:
            anchor_file = sorted(all_files)[0]
        
        reference_files = [f for f in all_files if f != anchor_file]
        if not reference_files:
            return None, 0.0

        with torch.no_grad():
            anchor_img = self.transform(Image.open(os.path.join(folder_path, anchor_file)).convert('RGB')).unsqueeze(0).to(self.device)
            anchor_embedding = self.model(anchor_img)

            best_similarity = -1.0
            most_similar_image = None

            for ref_name in reference_files:
                ref_img = self.transform(Image.open(os.path.join(folder_path, ref_name)).convert('RGB')).unsqueeze(0).to(self.device)
                ref_embedding = self.model(ref_img)

                similarity = torch.mm(anchor_embedding, ref_embedding.t()).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    most_similar_image = ref_name

        return most_similar_image, best_similarity