# ai_image_detector
Making an AI image detector with a friend for final year project.


# 🔍 AI Image Detector

A web-based tool that attempts to detect whether an image is **AI-generated** or **real/authentic** using image forensics and a self-learning ML model.

> ⚠️ **Work in progress** — the current detection is not perfect and may misclassify images. Accuracy improves the more you label. Built as a learning project with the help of AI (Claude by Anthropic).

---

## 📁 Project Structure

```
ai-image-detector/
├── app.py                  # Flask backend + feature extraction + classifier
├── requirements.txt        # Python dependencies
├── README.md
├── data/
│   ├── training_data.json  # Saved labeled images (auto-created)
│   └── model.pkl           # Trained ML model (created after retraining)
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles + Tailwind overrides
│   └── js/
│       └── main.js         # Frontend JavaScript
└── templates/
    └── index.html          # HTML (Tailwind CSS)
```

---

## 🚀 Getting Started

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

---

## 🧠 How It Works

### Detection
The app extracts 8 forensic signals from each image and scores them:

| Signal | What it detects |
|---|---|
| **Laplacian Variance** | AI images are unnaturally smooth at pixel level |
| **Color Smoothness** | Diffusion models produce even, perfect gradients |
| **Saturation Uniformity** | AI tends to over-saturate colors evenly |
| **Pixel Entropy** | Real photos have richer, messier tonal detail |
| **Edge Density** | AI images often lack fine structural edges |
| **JPEG Block Artefacts** | Real camera photos show compression marks |
| **Chromatic Aberration** | Real lenses distort color at edges; AI doesn't |
| **Resolution Pattern** | Common AI output sizes (512, 1024, 1536px) are flagged |

### Self-Learning
1. Analyze any image — starts with heuristic rules
2. Click **✅ Yes** or **❌ No** after each prediction to label it
3. Once you have 4+ samples, click **🧠 Retrain Model**
4. The app switches to an ML model (Random Forest) trained on your data
5. The more you label, the more accurate it gets

---

## 🛠️ Tech Stack

- **Backend** — Python, Flask
- **ML** — scikit-learn (Random Forest)
- **Image processing** — Pillow, NumPy
- **Frontend** — Tailwind CSS, Vanilla JavaScript
- **Storage** — JSON (training data), Pickle (model)

---

## 📦 Requirements

```
flask
pillow >= 10.0.0
numpy
scikit-learn
```

---

## ⚠️ Limitations

- Heuristic detection is not reliable on its own — label as many images as possible to improve the ML model
- Heavily edited real photos or AI images with added noise may confuse the detector
- The ML model is only as good as the data you provide