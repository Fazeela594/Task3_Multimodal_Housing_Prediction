import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import HousePriceModel

# -------------------------
# BASE DIRECTORY
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# Load Model
# -------------------------
model = HousePriceModel()

model_path = os.path.join(BASE_DIR, "model.pth")

model.load_state_dict(torch.load(model_path))

model.eval()

print("Model loaded successfully ✔")

# -------------------------
# Image Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------
# Load Image
# -------------------------
image_path = os.path.join(BASE_DIR, "images", "house1.jpg")

image = Image.open(image_path).convert("RGB")

image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)

# -------------------------
# Tabular Input
# -------------------------
# area = 1200
# rooms = 3

tabular = torch.tensor([[1200, 3]], dtype=torch.float32)

# -------------------------
# Prediction
# -------------------------
with torch.no_grad():

    prediction = model(image, tabular)

print("Predicted House Price:", prediction.item())