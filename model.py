import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class VGG16FineTuned(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def print_model_layers(self):
        print("VGG16 Architecture:\n", self.model)

import os
from torchvision import datasets

def load_model_and_classes(device):
    # Point to the same folder you used during training
    train_dir = "C:/Users/rohan/Documents/rishi/LY project/vgg16/dataset/Train"

    # Use ImageFolder to get class names
    dataset = datasets.ImageFolder(root=train_dir)
    classes = dataset.classes  # e.g., ['Fake', 'Real', 'OtherClass', ...]

    model = VGG16FineTuned(num_classes=len(classes))
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.to(device).eval()

    return model, classes
