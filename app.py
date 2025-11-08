# app.py - LOCAL VERSION (NO NGROK)
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
import base64

# -------------------------------
# FLASK SETUP
# -------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
MODEL_PATH = 'emotion_model_vortex.h5'
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found!")
    exit()
MODEL = load_model(MODEL_PATH)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

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
            name TEXT,
            image_path TEXT,
            emotion TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
init_db()

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
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (48, 48))
    norm = resized.astype('float32') / 255.0
    return np.expand_dims(norm, axis=[-1, 0])

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    file = request.files['image']
    if file and name:
        filename = f"{int(datetime.now().timestamp())}_{name}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = cv2.imread(filepath)
        pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
        idx = np.argmax(pred)
        emotion = EMOTIONS[idx]
        conf = float(pred[idx])
        save_to_db(name, f"static/uploads/{filename}", emotion, conf)
        return jsonify({
            'emotion': emotion,
            'confidence': round(conf, 3),
            'image_url': f"static/uploads/{filename}"
        })
    return jsonify({'error': 'Failed'}), 400

@app.route('/webcam', methods=['POST'])
def webcam():
    name = request.json['name']
    data = request.json['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    filename = f"webcam_{int(datetime.now().timestamp())}_{name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, img)
    pred = MODEL.predict(preprocess_image(img), verbose=0)[0]
    idx = np.argmax(pred)
    emotion = EMOTIONS[idx]
    conf = float(pred[idx])
    save_to_db(name, f"static/uploads/{filename}", emotion, conf)
    return jsonify({
        'emotion': emotion,
        'confidence': round(conf, 3),
        'image_url': f"static/uploads/{filename}"
    })

# -------------------------------
# RUN APP
# -------------------------------
if __name__ != '__main__':
    from gevent import monkey
    monkey.patch_all()
    
if __name__ == '__main__':
    print("Emotion Vortex running at http://localhost:5000")
    print("Press Ctrl+C to stop")
    app.run(host='127.0.0.1', port=5000, debug=True)
    
    # ... (your existing code)



