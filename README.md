# ai_image_detector
Making an AI image detector with a friend for final year project.


# 🔍 AI Image Detector

Detect whether an image is **AI-generated** or **real/authentic** using forensic heuristics.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python app.py
```

### 3. Open in your browser
```
http://localhost:5000
```

Drop or select any image and hit **Analyze Image** — results appear in seconds.

---

## 🧠 How It Works

The detector extracts **8 forensic signal types** and scores each:

| Signal | What it measures |
|---|---|
| **Laplacian Variance** | High-frequency noise — AI images are unnaturally smooth |
| **Color Smoothness** | Pixel-to-pixel gradient transitions |
| **Saturation uniformity** | AI often over-saturates evenly |
| **Pixel Entropy** | Richness of tonal information |
| **Edge Density** | Structural detail and sharp boundaries |
| **JPEG Block Artefacts** | Real camera photos show compression marks |
| **Chromatic Aberration** | Lens distortions absent in AI renders |
| **Resolution Pattern** | Matches standard AI model output sizes |

---

## 📁 Project Structure

```
ai-image-detector/
├── app.py              # Flask backend + feature extraction + classifier
├── requirements.txt    # Python dependencies
├── README.md
└── templates/
    └── index.html      # Frontend UI
```

---

## ⚠️ Limitations

- This uses **heuristic analysis**, not a trained ML model.
- Accuracy is high for typical AI art (Midjourney, DALL-E, Stable Diffusion) and natural photography.
- Heavily edited real photos or AI images with added noise may confuse the detector.
- For production accuracy, consider fine-tuning a CNN (e.g. EfficientNet) on a labelled dataset.

---

## 🛠️ Extending It

To add a trained ML model:

```python
# In app.py, replace classify_image() with:
import torch
from torchvision import transforms, models

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

Then call `model(tensor)` and combine with existing heuristics for best results.