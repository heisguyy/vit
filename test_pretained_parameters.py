"""Script to verify our implementation."""
from PIL import Image
import torch
from torchvision import transforms

from model import ViTForImageClassification, ViTConfig

# Define image preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

# Load and initialize model
model = ViTForImageClassification(ViTConfig())
loaded_state_dict = torch.load('vit_model.pth', weights_only=True)
model.load_state_dict(loaded_state_dict)
model.eval()

# Load and preprocess image
image = Image.open("n01443537_goldfish.JPEG").convert("RGB")
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

# Forward pass
with torch.no_grad():
    result = model(image_tensor)

probabilities = torch.nn.functional.softmax(result.squeeze(0), dim=0)
predicted_idx = torch.argmax(probabilities).item()
print(predicted_idx)
# Output: 1
# looking at the labelling mapping at
# https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/,
# 1 is goldfish
