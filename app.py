# app.py - Emotion Vortex (Production-Ready for Render)
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
import base64
from PIL import Image, UnidentifiedImageError
import io
from mtcnn import MTCNN

# -------------------------------
# FLASK SETUP
# -------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# LOAD MODEL (with error handling)
# -------------------------------
MODEL_PATH = 'emotion_model_vortex.h5'
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found! Place it in the project root.")
    exit(1)

try:
    MODEL = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# -------------------------------
# FACE DETECTOR (MTCNN)
# -------------------------------
try:
    face_detector = MTCNN()
    print("MTCNN face detector initialized.")
except Exception as e:
    print(f"MTCNN init failed: {e}. Falling back to full-image mode.")
    face_detector = None

# -------------------------------
# DATABASE
# -------------------------------
DB_PATH = 'emotions.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(name, image_path, emotion, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO detections (name, image_path, emotion, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (name, image_path, emotion, confidence, timestamp)
    )
    conn.commit()
    conn.close()

# -------------------------------
# IMAGE LOADING (Pillow → OpenCV)
# -------------------------------
def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Safely load image bytes using Pillow → convert to OpenCV BGR."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except UnidentifiedImageError:
        raise ValueError("Invalid or corrupted image file.")
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

# -------------------------------
# FACE DETECTION + PREPROCESS
# -------------------------------
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    1. Detect face with MTCNN (if available)
    2. Crop face with margin
    3. Resize to 48x48 grayscale
    4. Normalize and expand dims
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Face detection ---
    face_crop = img
    if face_detector:
        try:
            detections = face_detector.detect_faces(rgb)
            if detections:
                # Pick most confident face
                box = max(detections, key=lambda x: x['confidence'])['box']
                x, y, w, h = [max(0, int(v)) for v in box]
                margin = int(max(w, h) * 0.2)
                face_crop = img[
                    max(0, y - margin): y + h + margin,
                    max(0, x - margin): x + w + margin
                ]
        except Exception as e:
            print(f"MTCNN error: {e}. Using full image.")

    # --- Preprocess ---
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=[-1, 0])  # (1, 48, 48, 1)

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form.get('name', '').strip()
    file = request.files.get('image')

    if not name:
        return jsonify({'error': 'Name is required.'}), 400
    if not file or file.filename == '':
        return jsonify({'error': 'No image uploaded.'}), 400

    try:
        img_bytes = file.read()
        img = load_image_from_bytes(img_bytes)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Save full image
    filename = f"{int(datetime.now().timestamp())}_{name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    success = cv2.imwrite(filepath, img)
    if not success:
        return jsonify({'error': 'Failed to save image.'}), 500

    # Predict
    try:
        pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
        idx = int(np.argmax(pred))
        emotion = EMOTIONS[idx]
        conf = float(pred[idx])
    except Exception as e:
        return jsonify({'error': 'Model prediction failed.'}), 500

    # Save to DB
    save_to_db(name, f"static/uploads/{filename}", emotion, conf)

    return jsonify({
        'emotion': emotion,
        'confidence': round(conf, 3),
        'image_url': f"static/uploads/{filename}"
    })

@app.route('/webcam', methods=['POST'])
def webcam():
    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'Invalid JSON.'}), 400

    name = payload.get('name', '').strip()
    data_url = payload.get('image', '')

    if not name:
        return jsonify({'error': 'Name is required.'}), 400
    if not data_url.startswith('data:image'):
        return jsonify({'error': 'Invalid image data.'}), 400

    try:
        img_bytes = base64.b64decode(data_url.split(',')[1])
        img = load_image_from_bytes(img_bytes)
    except Exception as e:
        return jsonify({'error': 'Failed to decode image.'}), 400

    # Save
    filename = f"webcam_{int(datetime.now().timestamp())}_{name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, img)

    # Predict
    try:
        pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
        idx = int(np.argmax(pred))
        emotion = EMOTIONS[idx]
        conf = float(pred[idx])
    except Exception as e:
        return jsonify({'error': 'Model prediction failed.'}), 500

    save_to_db(name, f"static/uploads/{filename}", emotion, conf)

    return jsonify({
        'emotion': emotion,
        'confidence': round(conf, 3),
        'image_url': f"static/uploads/{filename}"
    })

# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'}), 200

# -------------------------------
# RUN APP (Gunicorn / Flask)
# -------------------------------
if __name__ == '__main__':
    init_db()
    print("Emotion Vortex running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    # Gunicorn (Render)
    init_db()
