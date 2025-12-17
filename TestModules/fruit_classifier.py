import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

class CustomCNN(nn.Module):

    def __init__(self, num_classes=30):
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

class OZnetClassifier:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['Apple_Gala', 'Apple_Golden Delicious', 'Avocado', 'Banana', 'Berry', 'Burmese Grape', 'Carambola', 'Date Palm', 'Dragon', 'Elephant Apple', 'Grape', 'Green Coconut', 'Guava', 'Hog Plum', 'Kiwi', 'Lichi', 'Malta', 'Mango Golden Queen', 'Mango_Alphonso', 'Mango_Amrapali', 'Mango_Bari', 'Mango_Himsagar', 'Olive', 'Orange', 'Palm', 'Persimmon', 'Pineapple', 'Pomegranate', 'Watermelon', 'White Pear']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.model = CustomCNN(num_classes=len(self.class_names))
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            return self.class_names[predicted_idx.item()], confidence.item()
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None, 0.0