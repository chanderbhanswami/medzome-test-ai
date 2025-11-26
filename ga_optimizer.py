import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, '../augmented_data')
IMG_HEIGHT = 384
IMG_WIDTH = 128
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# GA SETTINGS
POPULATION_SIZE = 8
GENERATIONS_ARCH_SEARCH = 3  # Stage 1: Find best model
GENERATIONS_FINE_TUNE = 5    # Stage 2: Optimize parameters
EPOCHS_PER_EVAL = 3          # Quick check

# 🧬 SEARCH SPACE 1: MODEL ARCHITECTURE
ARCH_SPACE = ['mobilenetv2', 'efficientnetb0', 'densenet121', 'resnet50']

# 🧬 SEARCH SPACE 2: HYPERPARAMETERS
PARAM_SPACE = {
    'dense_units': [64, 128, 256, 512],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.001, 0.0005, 0.0001, 1e-5],
    'optimizer': ['adam', 'rmsprop']
}

def get_data_generators(model_type):
    # Different models need different preprocessing
    if model_type == 'mobilenetv2':
        preprocess = applications.mobilenet_v2.preprocess_input
    elif model_type == 'densenet121':
        preprocess = applications.densenet.preprocess_input
    elif model_type == 'resnet50':
        preprocess = applications.resnet50.preprocess_input
    else: # EfficientNet expects 0-255 usually, but keras impl handles it
        preprocess = applications.efficientnet.preprocess_input

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        validation_split=0.2,
        rotation_range=10,
        fill_mode='nearest'
    )
    train_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32, class_mode='binary', subset='training', verbose=0
    )
    val_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32, class_mode='binary', subset='validation', verbose=0
    )
    return train_gen, val_gen

def build_model(genes):
    """Builds model based on genes"""
    arch = genes['architecture']
    
    # 1. Select Base Model
    if arch == 'mobilenetv2':
        base_model = applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
    elif arch == 'efficientnetb0':
        base_model = applications.EfficientNetB0(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
    elif arch == 'densenet121':
        base_model = applications.DenseNet121(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
    elif arch == 'resnet50':
        base_model = applications.ResNet50(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
    
    base_model.trainable = False # Freeze

    # 2. Build Head
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(genes['dense_units'], activation='relu')(x)
    x = layers.Dropout(genes['dropout_rate'])(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    
    # 3. Optimizer
    if genes['optimizer'] == 'adam':
        opt = optimizers.Adam(learning_rate=genes['learning_rate'])
    else:
        opt = optimizers.RMSprop(learning_rate=genes['learning_rate'])
        
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_individual(genes):
    print(f"\n🧬 Testing: {genes['architecture'].upper()} | Units: {genes['dense_units']} | LR: {genes['learning_rate']}")
    
    try:
        train_gen, val_gen = get_data_generators(genes['architecture'])
        model = build_model(genes)
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_PER_EVAL,
            verbose=1
        )
        score = max(history.history['val_accuracy'])
        print(f"   ✅ Score: {score:.4%}")
        tf.keras.backend.clear_session()
        return score
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return 0.0

class TwoStageGA:
    def __init__(self):
        self.history = []
        self.best_arch = None

    def create_random(self, fixed_arch=None):
        genes = {k: random.choice(v) for k, v in PARAM_SPACE.items()}
        genes['architecture'] = fixed_arch if fixed_arch else random.choice(ARCH_SPACE)
        return genes

    def run(self):
        print("🚀 STAGE 1: ARCHITECTURE SEARCH")
        population = [self.create_random() for _ in range(POPULATION_SIZE)]
        
        for gen in range(GENERATIONS_ARCH_SEARCH):
            print(f"\n--- Gen {gen+1} (Arch Search) ---")
            scores = []
            for ind in population:
                acc = evaluate_individual(ind)
                scores.append((ind, acc))
                self.history.append({**ind, 'accuracy': acc, 'stage': 1})
            
            scores.sort(key=lambda x: x[1], reverse=True)
            print(f"🏆 Best: {scores[0][1]:.4%} ({scores[0][0]['architecture']})")
            
            # Evolution
            top_half = [x[0] for x in scores[:POPULATION_SIZE//2]]
            new_pop = []
            while len(new_pop) < POPULATION_SIZE:
                # Simple mutation-based reproduction for stage 1
                parent = random.choice(top_half)
                child = parent.copy()
                if random.random() > 0.3: # 70% chance to change params
                    key = random.choice(list(PARAM_SPACE.keys()))
                    child[key] = random.choice(PARAM_SPACE[key])
                if random.random() > 0.5: # 50% chance to switch architecture
                    child['architecture'] = random.choice(ARCH_SPACE)
                new_pop.append(child)
            population = new_pop

        # Select Winner
        scores.sort(key=lambda x: x[1], reverse=True)
        self.best_arch = scores[0][0]['architecture']
        print(f"\n🌟 WINNING ARCHITECTURE: {self.best_arch.upper()}")
        
        print("\n🚀 STAGE 2: HYPERPARAMETER TUNING")
        # Re-initialize population locked to winning architecture
        population = [self.create_random(fixed_arch=self.best_arch) for _ in range(POPULATION_SIZE)]
        
        for gen in range(GENERATIONS_FINE_TUNE):
            print(f"\n--- Gen {gen+1} (Fine Tuning) ---")
            scores = []
            for ind in population:
                acc = evaluate_individual(ind)
                scores.append((ind, acc))
                self.history.append({**ind, 'accuracy': acc, 'stage': 2})
                
            scores.sort(key=lambda x: x[1], reverse=True)
            print(f"🏆 Best: {scores[0][1]:.4%}")
            
            top_half = [x[0] for x in scores[:POPULATION_SIZE//2]]
            new_pop = []
            while len(new_pop) < POPULATION_SIZE:
                p1, p2 = random.sample(top_half, 2)
                child = {k: p1[k] if random.random() > 0.5 else p2[k] for k in PARAM_SPACE.keys()}
                child['architecture'] = self.best_arch # Lock arch
                
                # Mutate
                if random.random() > 0.2:
                    key = random.choice(list(PARAM_SPACE.keys()))
                    child[key] = random.choice(PARAM_SPACE[key])
                new_pop.append(child)
            population = new_pop

        # Save
        df = pd.DataFrame(self.history)
        df.to_csv('ga_twostage_results.csv', index=False)
        print(f"\n✅ DONE. Best config saved to CSV.")

if __name__ == '__main__':
    if not os.path.exists(TRAIN_DIR):
        print("❌ Run augment_data.py first!")
    else:
        ga = TwoStageGA()
        ga.run()