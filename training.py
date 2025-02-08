"""Training script for ViT."""
import os
from dataclasses import dataclass

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from model import ViTForImageClassification, ViTConfig

# Initialize wandb for experiment tracking
os.makedirs("models", exist_ok=True)
wandb.init(project="font-classification-vit")

# Set device
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4


# Initialize model
@dataclass
class FontViTConfig:  # pylint: disable=too-many-instance-attributes
    """All the configuration parameters for the ViT model"""

    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 256
    image_size: int = 128
    initializer_range: float = 0.02
    intermediate_size: int = 1024
    layer_norm_eps: float = 1e-12
    num_attention_heads: int = 8
    num_channels: int = 3
    num_hidden_layers: int = 8
    patch_size: int = 16
    num_labels: int = 62


model = ViTForImageClassification(FontViTConfig()).to(device)

# Data transforms
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Load datasets
train_dataset = datasets.ImageFolder(root="fonts/train", transform=transform)
val_dataset = datasets.ImageFolder(root="fonts/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# Training loop
best_val_acc = 0 # pylint: disable=invalid-name
for epoch in range(EPOCHS):
    model.train()
    training_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    training_loss /= len(train_loader)
    training_accuracy = 100.0 * correct / total
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100.0 * correct / total

    # Log metrics
    wandb.log(
        {
            "epoch": epoch,
            "train_loss": training_loss,
            "train_acc": training_accuracy,
            "val_loss": val_loss,
            "val_acc": val_accuracy,
        }
    )

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {training_loss:.4f}, Train Acc: {training_accuracy:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save best model
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), "models/best_model.pth")

print("Training completed!")
wandb.finish()
