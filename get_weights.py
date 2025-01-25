"""Script to get weights from the huggingface and save the archictecture"""
# Load model directly
import torch
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)
with open("vit_architecture.txt", "w", encoding="utf-8") as f:
    f.write(str(model))
torch.save(model.state_dict(), "vit_model.pth")
