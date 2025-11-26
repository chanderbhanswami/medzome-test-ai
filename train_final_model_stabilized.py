import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import tf2onnx
import onnx
import json

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, '../augmented_data')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, '../models/medzome_final_model.keras')
ONNX_SAVE_PATH = os.path.join(BASE_DIR, '../models/medzome_final_model.onnx')
HISTORY_PATH = os.path.join(BASE_DIR, '../models/training_history.json')

IMG_HEIGHT = 384
IMG_WIDTH = 128
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
BATCH_SIZE = 32

# Phase 1 (Head only)
EPOCHS_PHASE_1 = 10
LR_PHASE_1 = 0.001
# Phase 2 (Fine Tuning)
EPOCHS_PHASE_2 = 20
LR_PHASE_2 = 1e-5  # Very slow learning to prevent crashing

# ============================================================================
# 2. DATA
# ============================================================================
def get_data_generators():
    # MobileNetV2 expects inputs in range [-1, 1]
    preprocess_func = applications.mobilenet_v2.preprocess_input
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        validation_split=0.2,
        rotation_range=10,       # Rotated strips
        width_shift_range=0.1,   # Off-center strips
        height_shift_range=0.1,
        zoom_range=0.1,          # Camera zoom variance
        fill_mode='nearest'
    )
    
    print(f"Loading data from: {TRAIN_DIR}")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='binary', subset='training', shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='binary', subset='validation', shuffle=False
    )
    return train_gen, val_gen

# ============================================================================
# 3. TRAINING
# ============================================================================
def main():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: {TRAIN_DIR} not found. Run augment_data.py first!")
        return

    train_gen, val_gen = get_data_generators()
    
    # Class Weights (Handle any remaining imbalance)
    labels = train_gen.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(labels), y=labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # --- PHASE 1: TRAIN HEAD ONLY ---
    print("\n🛡️  PHASE 1: Training Classifier Head (Base Frozen)...")
    
    base_model = applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'
    )
    base_model.trainable = False # Freeze completely

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output_class")(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_PHASE_1),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_gen,
        epochs=EPOCHS_PHASE_1,
        validation_data=val_gen,
        class_weight=class_weights_dict,
        verbose=1
    )

    # --- PHASE 2: FINE TUNING ---
    print("\n🔓 PHASE 2: Fine-Tuning Entire Model (Low LR)...")
    
    base_model.trainable = True # Unfreeze everything
    
    # Recompile with very low Learning Rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_PHASE_2),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    # Ensure models dir exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE_2,
        validation_data=val_gen,
        class_weight=class_weights_dict,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save & Export
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model Saved: {MODEL_SAVE_PATH}")
    
    # History
    with open(HISTORY_PATH, 'w') as f:
        # Convert values to float for JSON serialization
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(hist_dict, f)

    # ONNX
    try:
        spec = (tf.TensorSpec((None, IMG_HEIGHT, IMG_WIDTH, 3), tf.float32, name="input_image"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        onnx.save(model_proto, ONNX_SAVE_PATH)
        print(f"✅ ONNX Exported: {ONNX_SAVE_PATH}")
    except Exception as e:
        print(f"❌ ONNX Failed: {e}")

if __name__ == '__main__':
    main()