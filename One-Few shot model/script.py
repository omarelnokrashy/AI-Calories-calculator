import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from collections import defaultdict
import random
import math
import os

# ==========================================
# 1. CLASS DEFINITIONS (From your code)
# ==========================================

class EpisodicSampler:
    def __init__(self, labels, n_way, k_shot, q_query):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            self.label_to_indices[label].append(i)

    def sample_episode(self):
        classes = random.sample(list(self.label_to_indices.keys()), self.n_way)
        support, query = [], []
        query_labels = []

        for proto_id, cls in enumerate(classes):
            indices = self.label_to_indices[cls]
            num_available = len(indices)

            # Robustness check: if a class has fewer samples than needed, skip it or error
            if num_available < self.k_shot + 1:
                # In a real script, you might retry or filter these classes out beforehand
                raise ValueError(f"Class {cls} has too few samples ({num_available}) for k_shot={self.k_shot}")

            max_q = min(self.q_query, num_available - self.k_shot)
            selected = random.sample(indices, self.k_shot + max_q)

            support.extend(selected[:self.k_shot])
            query.extend(selected[self.k_shot:])
            query_labels.extend([proto_id] * max_q)

        return support, query, query_labels

class SwinProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Ensure we load the same architecture structure as training
        backbone = models.swin_t(weights=None) # Weights=None because we load custom weights later
        backbone.head = nn.Identity()
        self.encoder = backbone

    def forward(self, x):
        return self.encoder(x)

    def compute_loss(self, embeddings, query_labels, n_way, k_shot):
        support = embeddings[:n_way * k_shot]
        query = embeddings[n_way * k_shot:]

        prototypes = []
        for i in range(n_way):
            proto = support[i * k_shot:(i + 1) * k_shot].mean(dim=0)
            prototypes.append(F.normalize(proto, dim=0))
        prototypes = torch.stack(prototypes)

        query = F.normalize(query, dim=1)

        # Metric scaling (temperature)
        logits = torch.matmul(query, prototypes.t()) * 20.0
        
        loss = F.cross_entropy(logits, query_labels)
        acc = (logits.argmax(dim=1) == query_labels).float().mean()

        return loss, acc

# ==========================================
# 2. TEST FUNCTION
# ==========================================

def test_model(model_path, data_dir, n_way=5, k_shot=5, q_query=15, episodes=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Define Transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return

    print("Loading dataset...")
    dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    labels = [label for _, label in dataset.samples]
    print(f"Dataset loaded. Found {len(dataset)} images across {len(dataset.classes)} classes.")

    # 3. Initialize Sampler
    sampler = EpisodicSampler(labels, n_way, k_shot, q_query)

    # 4. Load Model
    print("Loading model...")
    model = SwinProtoNet()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        

        if 'model_state' in checkpoint:
            print("Found 'model_state' key in checkpoint. Loading from there...")
            model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:

            print("Found 'model_state_dict' key in checkpoint. Loading from there...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:

            model.load_state_dict(checkpoint)
            
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()
    accs = []

    print(f"Starting testing ({episodes} episodes)...")
    
    with torch.no_grad():
        for i in range(episodes):
            try:
                s_idx, q_idx, q_labels = sampler.sample_episode()
            except ValueError as e:
                continue

            idxs = s_idx + q_idx

            # Gather batch images
            images = torch.stack([dataset[j][0] for j in idxs]).to(device)
            q_labels = torch.tensor(q_labels).to(device)

            # Forward pass
            emb = model(images)
            _, acc = model.compute_loss(emb, q_labels, n_way, k_shot)
            
            accs.append(acc.item())

            if (i + 1) % 100 == 0:
                print(f"Episode {i+1}/{episodes}...")

    # 6. Calculate Statistics
    if len(accs) > 0:
        mean = sum(accs) / len(accs)
        std = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1))
        ci95 = 1.96 * std / math.sqrt(len(accs))

        print("\n" + "="*30)
        print(f"RESULTS: {n_way}-way {k_shot}-shot")
        print(f"Accuracy: {mean * 100:.2f}% ± {ci95 * 100:.2f}% (95% CI)")
        print("="*30)
    else:
        print("No valid episodes completed.")


if __name__ == "__main__":

    MODEL_PATH = "One-Few shot model\\SWIN_ProtoNet\\swin_protonet_hyperparameters.pth"     
    TEST_DATA_DIR = "One-Few shot model\\test"      
    N_WAY = 3                       
    K_SHOT = 5
    Q_QUERY = 3
    EPISODES = 100          
    
    test_model(
        model_path=MODEL_PATH, 
        data_dir=TEST_DATA_DIR, 
        n_way=N_WAY, 
        k_shot=K_SHOT, 
        q_query=Q_QUERY, 
        episodes=EPISODES
    )