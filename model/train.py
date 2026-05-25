import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from model import HousePriceModel

# -------------------------
# BASE DIRECTORY
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# Dataset Class
# -------------------------
class HouseDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Image path
        img_path = os.path.join(self.img_dir, row["image"])

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tabular data
        tabular = torch.tensor(
            [row["area"], row["rooms"]],
            dtype=torch.float32
        )

        # Price
        price = torch.tensor(
            [row["price"]],
            dtype=torch.float32
        )

        return image, tabular, price


# -------------------------
# Dataset Loading
# -------------------------
dataset = HouseDataset(
    csv_file=os.path.join(BASE_DIR, "data", "houses.csv"),
    img_dir=os.path.join(BASE_DIR, "images")
)

print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# -------------------------
# Model
# -------------------------
model = HousePriceModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Training started...")

# -------------------------
# Training Loop
# -------------------------
epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for img, tab, price in loader:

        # Forward
        output = model(img, tab)

        # Loss
        loss = criterion(output, price)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -------------------------
# Save Model
# -------------------------
model_path = os.path.join(BASE_DIR, "model.pth")

torch.save(model.state_dict(), model_path)

print("Training Done ✔")
print("Model saved at:", model_path)