"""
AI Image Detector - Flask Backend
Detects whether an uploaded image is AI-generated or real.
Supports user feedback + retraining via scikit-learn.

Usage:
    pip install -r requirements.txt
    python app.py
    Open http://localhost:5000 in your browser
"""

import os
import io
import base64
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'}

# Ensure data folder exists
os.makedirs("data", exist_ok=True)
DATA_FILE  = os.path.join("data", "training_data.json")
MODEL_FILE = os.path.join("data", "model.pkl")
RESAMPLE   = Image.Resampling.LANCZOS

# Load ML model if available
ml_model = None
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        ml_model = pickle.load(f)
    print("Trained model loaded from", MODEL_FILE)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_features(img):
    img_rgb = img.convert("RGB")
    arr     = np.array(img_rgb, dtype=np.float32)
    features = {}

    for i, ch in enumerate(['r', 'g', 'b']):
        channel = arr[:, :, i]
        features[f'{ch}_mean']  = float(channel.mean())
        features[f'{ch}_std']   = float(channel.std())
        features[f'{ch}_range'] = float(channel.max() - channel.min())

    gray = np.array(img_rgb.convert("L"), dtype=np.float32)
    lap  = _laplacian(gray)
    features['laplacian_var']      = float(np.var(lap))
    features['laplacian_mean_abs'] = float(np.mean(np.abs(lap)))
    features['entropy']            = float(_image_entropy(gray))
    features['color_smoothness']   = float(_color_smoothness(arr))

    edges = _sobel(gray)
    features['edge_density'] = float(np.mean(edges > 30))
    features['edge_mean']    = float(np.mean(edges))

    try:
        hsv = np.array(img_rgb.convert("HSV"), dtype=np.float32)
        features['sat_mean'] = float(hsv[:, :, 1].mean())
        features['sat_std']  = float(hsv[:, :, 1].std())
    except Exception:
        features['sat_mean'] = 128.0
        features['sat_std']  = 50.0

    features['block_artifact']    = float(_block_artifact_score(gray))
    features['chroma_aberration'] = float(_chroma_aberration(arr))

    w, h = img.size
    features['width']      = float(w)
    features['height']     = float(h)
    features['aspect']     = float(w / max(h, 1))
    features['megapixels'] = float(w * h / 1e6)

    return features


def features_to_vector(features):
    keys = sorted(features.keys())
    return np.array([features[k] for k in keys], dtype=np.float32)


def _laplacian(gray):
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    out    = np.zeros_like(gray)
    padded = np.pad(gray, 1, mode='reflect')
    for di in range(3):
        for dj in range(3):
            if kernel[di, dj] != 0:
                out += kernel[di, dj] * padded[di:di+gray.shape[0], dj:dj+gray.shape[1]]
    return out

def _sobel(gray):
    p = np.pad(gray, 1, mode='reflect')
    gx = (-p[:-2,:-2]-2*p[1:-1,:-2]-p[2:,:-2]+p[:-2,2:]+2*p[1:-1,2:]+p[2:,2:])
    gy = (-p[:-2,:-2]-2*p[:-2,1:-1]-p[:-2,2:]+p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:])
    return np.sqrt(gx**2 + gy**2)

def _image_entropy(gray):
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0,256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def _color_smoothness(arr):
    dy = np.abs(arr[1:,:,:] - arr[:-1,:,:]).mean()
    dx = np.abs(arr[:,1:,:] - arr[:,:-1,:]).mean()
    return float((dx + dy) / 2)

def _block_artifact_score(gray):
    h, w = gray.shape
    score, count = 0.0, 0
    for i in range(8, h, 8):
        score += float(np.abs(gray[i,:w//8*8] - gray[i-1,:w//8*8]).mean()); count += 1
    for j in range(8, w, 8):
        score += float(np.abs(gray[:h//8*8,j] - gray[:h//8*8,j-1]).mean()); count += 1
    return score / max(count, 1)

def _chroma_aberration(arr):
    return float(np.abs(arr[:,:,0] - arr[:,:,2]).std())


# ── Heuristic Classifier ──────────────────────────────────────────────────────

def heuristic_classify(features):
    ai_score = 0.0
    signals  = []

    lap = features['laplacian_var']
    if lap < 80:    ai_score += 0.20; signals.append(("AI",   f"Very low noise ({lap:.1f}) — AI smoothing"))
    elif lap < 300: ai_score += 0.08; signals.append(("AI",   f"Below-average noise ({lap:.1f})"))
    elif lap > 2000:ai_score -= 0.10; signals.append(("REAL", f"High noise ({lap:.1f}) — camera sensor"))

    sm = features['color_smoothness']
    if sm < 4.0:  ai_score += 0.18; signals.append(("AI",   f"Very smooth colors ({sm:.2f}) — diffusion model"))
    elif sm < 7.0:ai_score += 0.08; signals.append(("AI",   f"Smooth gradients ({sm:.2f})"))
    elif sm > 15: ai_score -= 0.08; signals.append(("REAL", f"High color variance ({sm:.2f})"))

    sat, ss = features['sat_mean'], features['sat_std']
    if sat > 180 and ss < 40: ai_score += 0.12; signals.append(("AI",   f"Uniform high saturation ({sat:.1f})"))
    elif sat < 60:            ai_score -= 0.05; signals.append(("REAL", f"Low saturation ({sat:.1f})"))

    ent = features['entropy']
    if ent > 7.6:  ai_score -= 0.10; signals.append(("REAL", f"High entropy ({ent:.2f}) — rich detail"))
    elif ent < 6.5:ai_score += 0.12; signals.append(("AI",   f"Low entropy ({ent:.2f})"))

    ed = features['edge_density']
    if ed < 0.03:  ai_score += 0.10; signals.append(("AI",   f"Very few edges ({ed*100:.1f}%)"))
    elif ed > 0.20:ai_score -= 0.08; signals.append(("REAL", f"Dense edges ({ed*100:.1f}%)"))

    bl = features['block_artifact']
    if bl > 5.0:  ai_score -= 0.08; signals.append(("REAL", f"JPEG artefacts ({bl:.2f}) — real photo"))
    elif bl < 1.0:ai_score += 0.06; signals.append(("AI",   f"No compression artefacts ({bl:.2f})"))

    ch = features['chroma_aberration']
    if ch > 30:  ai_score -= 0.07; signals.append(("REAL", f"Chromatic aberration ({ch:.1f}) — real lens"))
    elif ch < 8: ai_score += 0.07; signals.append(("AI",   f"No chromatic aberration ({ch:.1f})"))

    w, h = features['width'], features['height']
    if int(w) in {512,768,1024,1152,1216,1344,1536} and int(h) in {512,768,1024,1152,1216,1344,1536}:
        ai_score += 0.08; signals.append(("AI", f"Resolution {int(w)}x{int(h)} matches AI output sizes"))

    ai_score   = max(0.0, min(1.0, 0.5 + ai_score))
    confidence = max(10, min(99, int(round(abs(ai_score - 0.5) * 2 * 100))))
    label      = "AI Generated" if ai_score >= 0.5 else "Real / Authentic"

    return {"label": label, "confidence": confidence,
            "ai_probability": round(ai_score * 100, 1),
            "signals": signals, "method": "heuristic"}


# ── ML Classifier ─────────────────────────────────────────────────────────────

def ml_classify(features):
    if ml_model is None:
        return heuristic_classify(features)
    vec     = features_to_vector(features).reshape(1, -1)
    prob    = ml_model.predict_proba(vec)[0]
    ai_prob = float(prob[1])
    conf    = max(10, min(99, int(abs(ai_prob - 0.5) * 2 * 100)))
    label   = "AI Generated" if ai_prob >= 0.5 else "Real / Authentic"
    return {"label": label, "confidence": conf,
            "ai_probability": round(ai_prob * 100, 1),
            "signals": [("AI" if ai_prob >= 0.5 else "REAL",
                         f"ML model prediction (trained on {_training_count()} samples)")],
            "method": "ml"}


# ── Training helpers ──────────────────────────────────────────────────────────

def _training_count():
    if not os.path.exists(DATA_FILE):
        return 0
    with open(DATA_FILE) as f:
        return len(json.load(f))

def save_training_sample(features, label):
    data = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            data = json.load(f)
    data.append({"features": features, "label": label})
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

def retrain():
    global ml_model
    if not os.path.exists(DATA_FILE):
        return False, "No training data found."

    with open(DATA_FILE) as f:
        data = json.load(f)

    if len(data) < 4:
        return False, f"Need at least 4 labelled samples. You have {len(data)}."

    if len(set(d["label"] for d in data)) < 2:
        return False, "Need both AI and Real samples to train."

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = np.array([features_to_vector(d["features"]) for d in data])
    y = np.array([1 if d["label"] == "AI Generated" else 0 for d in data])

    model = Pipeline([("scaler", StandardScaler()),
                      ("clf",    RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    ml_model = model
    return True, f"Model trained successfully on {len(data)} samples!"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), RESAMPLE)

        features = extract_features(img)
        result   = ml_classify(features) if ml_model else heuristic_classify(features)

        result["features"] = {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in features.items()}

        preview = img.copy()
        preview.thumbnail((400, 400), RESAMPLE)
        buf = io.BytesIO()
        preview.save(buf, format="JPEG", quality=85)
        result["preview"]        = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        result["features_json"]  = json.dumps(result["features"])
        result["training_count"] = _training_count()
        result["model_active"]   = ml_model is not None

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    body = request.get_json()
    if not body or "features" not in body or "correct_label" not in body:
        return jsonify({"error": "Missing fields"}), 400

    label = body["correct_label"]
    if label not in ("AI Generated", "Real / Authentic"):
        return jsonify({"error": "Invalid label"}), 400

    save_training_sample(body["features"], label)
    return jsonify({"saved": True, "total": _training_count()})


@app.route("/retrain", methods=["POST"])
def retrain_route():
    ok, msg = retrain()
    return jsonify({"success": ok, "message": msg, "training_count": _training_count()})


@app.route("/stats")
def stats():
    return jsonify({"training_count": _training_count(), "model_active": ml_model is not None})


if __name__ == "__main__":
    print("\n AI Image Detector running at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)