import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2

# Ensure TensorFlow uses GPU efficiently
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set dataset path
dataset_path = 'D:/archive (1)/Gesture Image Data'

# Define parameters
image_size = (224, 224)
batch_size = 8  # Optimized for RTX 4050

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation'
)

num_classes = len(train_generator.class_indices)

# Load MobileNetV2
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Initially freeze all layers

# Gradual unfreezing strategy
def unfreeze_layers(model, num_unfreeze):
    for layer in model.layers[-num_unfreeze:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

unfreeze_layers(base_model, 30)  # Unfreeze last 30 layers

# Define model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

# Compile with Cosine Decay Learning Rate
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=1000, alpha=1e-5
)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_hand_sign_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train model
epochs = 50
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[model_checkpoint])


print("Training Complete. Best model saved as 'best_hand_sign_model.h5'")
