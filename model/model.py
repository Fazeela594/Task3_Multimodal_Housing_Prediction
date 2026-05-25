import torch
import torch.nn as nn
import torchvision.models as models

class HousePriceModel(nn.Module):
    def __init__(self, tabular_input_size=2):
        super().__init__()

        # 🖼️ Image Model (CNN - ResNet)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()   # last layer remove

        # 📊 Tabular model (area + rooms)
        self.tabular = nn.Sequential(
            nn.Linear(tabular_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 🔗 Combined model
        self.final = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # price output
        )

    def forward(self, image, tabular):
        img_features = self.cnn(image)
        tab_features = self.tabular(tabular)

        combined = torch.cat((img_features, tab_features), dim=1)
        return self.final(combined)