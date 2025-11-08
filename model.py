# model.py
# Standalone script to train the emotion detection model
# Dataset: FER-2013 (folders: train/ and test/)
# Model: CNN (3 Conv + 2 Dense)
# Output: emotion_model_vortex.h5

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ------------------- MODEL -------------------
def build_emotion_model():
    model = Sequential([
        Conv2D(32, (3,3), padding='same', input_shape=(48,48,1)), Activation('relu'),
        BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

        Conv2D(64, (3,3), padding='same'), Activation('relu'),
        BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

        Conv2D(128, (3,3), padding='same'), Activation('relu'),
        BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

        Flatten(),
        Dense(256), Activation('relu'), BatchNormalization(), Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------- DATA -------------------
IMG_SIZE = 48
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'train', target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale', batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    'test', target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale', batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# ------------------- TRAIN -------------------
model = build_emotion_model()
print("Starting training...")
model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    epochs=30,
    validation_data=test_gen,
    validation_steps=test_gen.samples // BATCH_SIZE
)

# ------------------- SAVE -------------------
os.makedirs("output", exist_ok=True)
model.save("output/emotion_model_vortex.h5")
print("Model saved to output/emotion_model_vortex.h5")
