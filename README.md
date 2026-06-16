# CIFAR-10 Image Classifier — ResNet-18 Transfer Learning

A PyTorch image classifier trained on CIFAR-10 using a pretrained ResNet-18 backbone.

---

## Quick Start

### 1. Clone / download this project

```bash
git clone <your-repo-url>
cd cifar10_classifier
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook Image_Classification_CIFAR10.ipynb
```

Run all cells top-to-bottom. CIFAR-10 data (~170 MB) will be downloaded automatically to `./data/` on first run.

---

## Project Structure

```
cifar10_classifier/
├── Image_Classification_CIFAR10.ipynb   # Main training & evaluation notebook
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── documentation.md                     # Architecture & design notes
└── (generated at runtime)
    ├── data/                            # CIFAR-10 dataset (auto-downloaded)
    ├── resnet18_cifar10.pth             # Saved model weights
    ├── sample_images.png                # Augmented training images grid
    ├── training_curves.png              # Loss & accuracy curves
    ├── confusion_matrix.png             # Test-set confusion matrix
    └── inference_samples.png            # Sample predictions on test images
```

---

## Running Inference on a Single Image

After training, load the saved model and run inference:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load('resnet18_cifar10.pth', map_location=device))
model.eval().to(device)

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

img = Image.open('your_image.jpg').convert('RGB')
tensor = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)
    conf, pred = probs.max(1)

print(f"Prediction : {CLASSES[pred.item()]}  ({conf.item()*100:.1f}% confidence)")
```

---

## Expected Results

| Metric              | Expected value (10 epochs, Adam, CosineAnnealingLR) |
|---------------------|-----------------------------------------------------|
| Test accuracy       | ~87–90%                                             |
| Training time (GPU) | ~5–8 min                                            |
| Training time (CPU) | ~60–90 min                                          |

---

## Hardware Notes

- **GPU recommended** (NVIDIA CUDA). The notebook auto-detects the device.
- On CPU, reduce `NUM_EPOCHS` to 3–5 for a quick smoke test.
- `pin_memory=True` and `num_workers=2` are set for GPU throughput. Set `NUM_WORKERS=0` on Windows if you encounter DataLoader multiprocessing errors.
