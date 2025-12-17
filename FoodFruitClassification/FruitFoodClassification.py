import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import random  # Added for random selection

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "FoodFruitClassification\\model_1.pth" 
TEST_IMAGE_FOLDER = "test"
CLASS_NAMES = ["Food", "Fruit"] 
IMAGE_SIZE = (224, 224)
NUM_SAMPLES = 12                       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=13, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        x = F.relu(self.conv7(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 3. UTILS
# ==========================================
def load_model(path, num_classes):
    print(f"Loading model from {path}...")
    model = CustomCNN(num_classes=num_classes)
    try:
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    model.to(DEVICE)
    model.eval()
    return model

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(model, image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None, None, None

    input_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return image, predicted_idx.item(), confidence.item()

# ==========================================
# 4. MAIN LOOP (RANDOMIZED)
# ==========================================
def run_test():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_IMAGE_FOLDER):
        print("Error: Check your MODEL_PATH and TEST_IMAGE_FOLDER.")
        return

    # Load Model
    model = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES))
    if model is None: return

    # Get all images
    all_files = [f for f in os.listdir(TEST_IMAGE_FOLDER) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    total_images = len(all_files)
    print(f"Found {total_images} total images.")

    if total_images == 0:
        print("No images found to test.")
        return

    # --- RANDOM SELECTION ---
    # Pick random samples (or all if less than NUM_SAMPLES)
    sample_count = min(NUM_SAMPLES, total_images)
    selected_files = random.sample(all_files, sample_count)
    
    print(f"Randomly selected {sample_count} images for visualization...\n")

    # Plotting setup
    plt.figure(figsize=(16, 10))
    cols = 4
    rows = (sample_count + cols - 1) // cols

    for i, img_file in enumerate(selected_files):
        img_path = os.path.join(TEST_IMAGE_FOLDER, img_file)
        
        # Predict
        original_img, pred_idx, conf = predict_image(model, img_path)
        
        if original_img is None: continue
        
        pred_label = CLASS_NAMES[pred_idx]
        print(f"[{i+1}/{sample_count}] {img_file} -> {pred_label} ({conf*100:.2f}%)")

        # Add to subplot
        plt.subplot(rows, cols, i + 1)
        plt.imshow(original_img)
        plt.title(f"{pred_label}\n{conf*100:.1f}%", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()