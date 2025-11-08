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

# -------------------------------------------------
# 1. TENSORFLOW OPTIMISATIONS (add at the very top)
# -------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # hide INFO/WARNING
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Optimise for Render’s shared CPU
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# -------------------------------------------------
# FLASK SETUP
# -------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------------------------
# 2. LOAD & WARM-UP MODEL (at startup)
# -------------------------------------------------
print("Loading model...")
MODEL_PATH = 'emotion_model_vortex.h5'
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found!")
    exit(1)

try:
    MODEL = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # ---- Warm-up with a dummy image ----
    dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
    _ = MODEL.predict(dummy, verbose=0)
    print("Model warmed up.")
except Exception as e:
    print(f"Model load failed: {e}")
    exit(1)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# -------------------------------------------------
# FACE DETECTOR
# -------------------------------------------------
try:
    face_detector = MTCNN()
    print("MTCNN face detector initialized.")
except Exception as e:
    print(f"MTCNN init failed: {e}")
    face_detector = None

# -------------------------------------------------
# DATABASE
# -------------------------------------------------
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

# -------------------------------------------------
# IMAGE LOADING
# -------------------------------------------------
def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except UnidentifiedImageError:
        raise ValueError("Invalid image file.")
    except Exception as e:
        raise ValueError(f"Image error: {e}")

# -------------------------------------------------
# PREPROCESS (face crop + 48×48)
# -------------------------------------------------
def preprocess_image(img: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_crop = img
    if face_detector:
        try:
            detections = face_detector.detect_faces(rgb)
            if detections:
                box = max(detections, key=lambda x: x['confidence'])['box']
                x, y, w, h = [max(0, int(v)) for v in box]
                margin = int(max(w, h) * 0.2)
                face_crop = img[
                    max(0, y - margin): y + h + margin,
                    max(0, x - margin): x + w + margin
                ]
        except Exception as e:
            print(f"MTCNN error: {e}")

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    resized = cv2.resize(gray, (48, 48))
    norm = resized.astype('float32') / 255.0
    return np.expand_dims(norm, axis=[-1, 0])

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form.get('name', '').strip()
    file = request.files.get('image')
    if not name or not file:
        return jsonify({'error': 'Name and image required.'}), 400

    try:
        img = load_image_from_bytes(file.read())
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    filename = f"{int(datetime.now().timestamp())}_{name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, img)

    try:
        pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
        idx = int(np.argmax(pred))
        emotion = EMOTIONS[idx]
        conf = float(pred[idx])
    except Exception as e:
        return jsonify({'error': 'Prediction failed.'}), 500

    save_to_db(name, f"static/uploads/{filename}", emotion, conf)
    return jsonify({
        'emotion': emotion,
        'confidence': round(conf, 3),
        'image_url': f"static/uploads/{filename}"
    })

@app.route('/webcam', methods=['POST'])
def webcam():
    payload = request.get_json()
    name = payload.get('name', '').strip()
    data_url = payload.get('image', '')
    if not name or not data_url.startswith('data:image'):
        return jsonify({'error': 'Invalid request.'}), 400

    try:
        img_bytes = base64.b64decode(data_url.split(',')[1])
        img = load_image_from_bytes(img_bytes)
    except Exception:
        return jsonify({'error': 'Failed to decode image.'}), 400

    filename = f"webcam_{int(datetime.now().timestamp())}_{name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, img)

    try:
        pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
        idx = int(np.argmax(pred))
        emotion = EMOTIONS[idx]
        conf = float(pred[idx])
    except Exception:
        return jsonify({'error': 'Prediction failed.'}), 500

    save_to_db(name, f"static/uploads/{filename}", emotion, conf)
    return jsonify({
        'emotion': emotion,
        'confidence': round(conf, 3),
        'image_url': f"static/uploads/{filename}"
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

# -------------------------------------------------
# STARTUP
# -------------------------------------------------
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    init_db()
