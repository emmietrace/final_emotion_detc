# app.py - TFLite Runtime Version (Render Optimized)
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import sqlite3
from datetime import datetime
import base64

# CHANGED: Use the lightweight runtime instead of full TensorFlow
import tflite_runtime.interpreter as tflite

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------
# LOAD TFLITE MODEL
# -------------------------------
# CHANGED: Load using tflite_runtime
interpreter = tflite.Interpreter(model_path="emotion_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Emotions for Mini-Xception model
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

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
def process_and_predict(image_path):
    try:
        # 1. Read Image
        img = cv2.imread(image_path)
        
        # 2. Face Detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
        else:
            roi_gray = gray

        # 3. Resize and Normalize
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype('float32') / 255.0
        
        # 4. Expand dims
        input_data = np.expand_dims(roi_gray, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)

        # 5. Predict using the lightweight interpreter
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        idx = np.argmax(output_data)
        emotion = EMOTIONS[idx]
        confidence = float(output_data[idx] * 100)

        return emotion, confidence
    except Exception as e:
        print(f"Error: {e}")
        return None, None

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

    filename = f"{int(datetime.now().timestamp())}_{name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    emotion, conf = process_and_predict(filepath)

    if emotion:
        save_to_db(name, f"static/uploads/{filename}", emotion, conf)
        return jsonify({
            'emotion': emotion,
            'confidence': round(conf, 1),
            'image_url': f"static/uploads/{filename}"
        })
    else:
        return jsonify({'error': 'Processing failed'}), 400

@app.route('/webcam', methods=['POST'])
def webcam():
    data = request.get_json()
    name = data.get('name', '').strip()
    img_data = data.get('image','')

    if not name or not img_data.startswith('data:image'):
        return jsonify({'error': 'Invalid data'}), 400

    try:
        header, encoded = img_data.split(",", 1)
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except: return jsonify({'error': 'Decode failed'}), 400

    filename = f"webcam_{int(datetime.now().timestamp())}_{name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, img)

    emotion, conf = process_and_predict(filepath)

    if emotion:
        save_to_db(name, f"static/uploads/{filename}", emotion, conf)
        return jsonify({
            'emotion': emotion,
            'confidence': round(conf, 1),
            'image_url': f"static/uploads/{filename}"
        })
    else:
        return jsonify({'error': 'No face detected'}), 400

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
