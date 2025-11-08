# app.py - TFLite Version (Render Free Tier Ready)
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import sqlite3
from datetime import datetime
import base64
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# LOAD TFLITE MODEL
# -------------------------------
interpreter = tflite.Interpreter(model_path='emotion_model_vortex.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded")

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
            id INTEGER PRIMARY KEY,
            name TEXT, image_path TEXT, emotion TEXT, confidence REAL, timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(name, image_path, emotion, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO detections (name, image_path, emotion, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (name, image_path, emotion, confidence, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

# -------------------------------
# IMAGE PROCESSING
# -------------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=[0, -1])  # (1, 48, 48, 1)

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
    if not name or not file: return jsonify({'error': 'Missing data'}), 400

    try:
        img = np.array(Image.open(file).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except: return jsonify({'error': 'Invalid image'}), 400

    filename = f"{int(datetime.now().timestamp())}_{name}.jpg"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), img)

    input_data = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(pred))
    emotion = EMOTIONS[idx]
    conf = float(pred[idx])

    save_to_db(name, f"static/uploads/{filename}", emotion, conf)
    return jsonify({
        'emotion': emotion,
        'confidence': round(conf, 3),
        'image_url': f"static/uploads/{filename}"
    })

@app.route('/webcam', methods=['POST'])
def webcam():
    data = request.get_json()
    name = data.get('name', '').strip()
    img_data = data.get('image','')

    if not name or not img_data.startswith('data:image'):
        return jsonify({'error': 'Invalid'}), 400

    try:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except: return jsonify({'error': 'Decode failed'}), 400

    filename = f"webcam_{int(datetime.now().timestamp())}_{name}.jpg"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), img)

    input_data = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(pred))
    emotion = EMOTIONS[idx]
    conf = float(pred[idx])

    save_to_db(name, f"static/uploads/{filename}", emotion, conf)
    return jsonify({
        'emotion': emotion,
        'confidence': round(conf, 3),
        'image_url': f"static/uploads/{filename}"
    })

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
else:
    init_db()
