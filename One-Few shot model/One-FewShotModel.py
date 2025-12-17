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

def run_siamese_test(model_path, folder_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load the Model
    model = SwinProtoNet()
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different saving formats
    state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    anchor_file = next((f for f in all_files if 'anchor' in f.lower()), None)
    if not anchor_file:

        anchor_file = sorted(all_files)[0]
    
    reference_files = [f for f in all_files if f != anchor_file]


    with torch.no_grad():
        anchor_img = transform(Image.open(os.path.join(folder_path, anchor_file)).convert('RGB')).unsqueeze(0).to(device)
        anchor_embedding = model(anchor_img)
        best_similarity = -1.0
        most_similar_image = None
        print(f"Comparing Anchor: {anchor_file} against {len(reference_files)} references...\n")
        for ref_name in reference_files:
            ref_img = transform(Image.open(os.path.join(folder_path, ref_name)).convert('RGB')).unsqueeze(0).to(device)
            ref_embedding = model(ref_img)
            
            similarity = torch.mm(anchor_embedding, ref_embedding.t()).item()
            print(f"Similarity with {ref_name}: {similarity:.4f}")

            if similarity > best_similarity:
                best_similarity = similarity
                most_similar_image = ref_name

    print("-" * 30)
    print(f"RESULT: The most similar image is '{most_similar_image}'")
    print(f"Confidence (Similarity Score): {best_similarity:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    MODEL_PATH = "One-Few shot model\\SWIN_ProtoNet\\swin_protonet_hyperparameters.pth"
    TEST_FOLDER = "Test Cases Structure\\Siamese Case II Test" 
    
    run_siamese_test(MODEL_PATH, TEST_FOLDER)