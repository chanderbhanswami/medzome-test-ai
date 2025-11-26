import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import logging
import time
import traceback

# ============================================================================
# 1. CONFIGURATION & SEARCH SPACE
# ============================================================================

# GA Settings
POPULATION_SIZE = 10      # Individuals per generation
GENERATIONS = 5           # How many evolution cycles to run
EPOCHS_PER_EVAL = 3       # Keep this low (3-5) for speed during search
SIMULATION_MODE = False   # Set True to TEST the script logic without training

# Dynamically find the path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, '../augmented_data')

# Image Dimensions
IMG_HEIGHT = 384
IMG_WIDTH = 128
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# 🧬 THE GENE POOL (Hyperparameter Search Space)
SEARCH_SPACE = {
    'architecture': [
        'mobilenetv2',       # Needs preprocess_input (-1 to 1)
        'mobilenetv3small',  # Internal Rescaling (Expects 0-255)
        'efficientnetb0',    # Internal Rescaling (Expects 0-255)
        'densenet121',       # Needs preprocess_input
        'nasnetmobile',      # Needs preprocess_input
        'custom_cnn'         # Needs rescale=1./255
    ],
    'learning_rate': [0.001, 0.0005, 0.0001, 1e-5],
    'optimizer': ['adam', 'rmsprop', 'sgd_momentum'],
    'batch_size': [16, 32],
    'dropout_rate': [0.2, 0.3, 0.5],
    'dense_units': [64, 128, 256],
    'augmentation': ['light', 'medium', 'heavy']
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# ============================================================================
# 2. MODEL FACTORY
# ============================================================================

def get_data_config(arch_name):
    """
    Returns the correct data generator configuration for each architecture.
    - preprocess_func: Keras function to normalize data
    - use_rescale: Boolean, whether to simply divide by 255
    """
    if arch_name == 'mobilenetv2':
        return applications.mobilenet_v2.preprocess_input, False
    elif arch_name == 'densenet121':
        return applications.densenet.preprocess_input, False
    elif arch_name == 'nasnetmobile':
        return applications.nasnet.preprocess_input, False
    elif arch_name == 'custom_cnn':
        return None, True  # Custom CNN needs manual 1/255 scaling
    else:
        # MobileNetV3 and EfficientNet have INTERNAL rescaling layers.
        # They expect raw [0-255] input.
        return None, False

def build_model(genes):
    """Assembles a model based on the 'genes' (parameters) provided"""
    arch = genes['architecture']
    dropout = genes['dropout_rate']
    units = genes['dense_units']
    
    # --- 1. Base Model Selection ---
    base_model = None
    
    # Common arguments for transfer learning models
    # We allow variable input shapes where possible, though some models warn if not 224x224
    kwargs = {'input_shape': INPUT_SHAPE, 'include_top': False, 'weights': 'imagenet'}
    
    if arch == 'mobilenetv2':
        base_model = applications.MobileNetV2(**kwargs)
    elif arch == 'mobilenetv3small':
        base_model = applications.MobileNetV3Small(**kwargs)
    elif arch == 'efficientnetb0':
        base_model = applications.EfficientNetB0(**kwargs)
    elif arch == 'densenet121':
        base_model = applications.DenseNet121(**kwargs)
    elif arch == 'nasnetmobile':
        base_model = applications.NASNetMobile(**kwargs)

    # --- 2. Construct Architecture ---
    if base_model:
        # Transfer Learning approach
        base_model.trainable = False 
        
        inputs = tf.keras.Input(shape=INPUT_SHAPE)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs, outputs)
        
    else:
        # Custom CNN approach
        inputs = tf.keras.Input(shape=INPUT_SHAPE)
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs, outputs)

    # --- 3. Compile with Optimizer ---
    lr = genes['learning_rate']
    opt_name = genes['optimizer']
    
    if opt_name == 'adam': 
        optimizer = optimizers.Adam(learning_rate=lr)
    elif opt_name == 'rmsprop': 
        optimizer = optimizers.RMSprop(learning_rate=lr)
    elif opt_name == 'sgd_momentum': 
        optimizer = optimizers.SGD(learning_rate=lr, momentum=0.9)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================================================================
# 3. GENETIC ALGORITHM ENGINE
# ============================================================================

class GeneticOptimizer:
    def __init__(self, search_space, population_size, generations):
        self.search_space = search_space
        self.pop_size = population_size
        self.generations = generations
        self.history = [] 

    def create_individual(self):
        """Spawns a random individual"""
        return {k: random.choice(v) for k, v in self.search_space.items()}

    def initialize_population(self):
        return [self.create_individual() for _ in range(self.pop_size)]

    def evaluate_fitness(self, individual, generation_idx, ind_idx):
        """Calculates fitness (Validation Accuracy)"""
        
        if SIMULATION_MODE:
            # Fake simulation delay and score
            time.sleep(0.1)
            return random.uniform(0.7, 0.99)

        print(f"\n🧬 Gen {generation_idx+1} | Ind {ind_idx+1}: {individual['architecture'].upper()}")
        print(f"   Opt: {individual['optimizer']} | LR: {individual['learning_rate']} | Drop: {individual['dropout_rate']}")

        # 1. Setup Data Generators
        preprocess_func, use_rescale = get_data_config(individual['architecture'])
        
        if preprocess_func:
            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func, validation_split=0.2)
        elif use_rescale:
            train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        else:
            train_datagen = ImageDataGenerator(validation_split=0.2)

        # Apply Augmentation Gene (CRITICAL FIX: Explicit Tuples for zoom_range)
        if individual['augmentation'] == 'medium':
            train_datagen.rotation_range = 10
            # FIX: Explicit tuple (0.9, 1.1) prevents 'float is not subscriptable' error
            train_datagen.zoom_range = (0.9, 1.1) 
        elif individual['augmentation'] == 'heavy':
            train_datagen.rotation_range = 20
            # FIX: Explicit tuple (0.8, 1.2)
            train_datagen.zoom_range = (0.8, 1.2)
            train_datagen.width_shift_range = 0.1
            train_datagen.height_shift_range = 0.1
        
        try:
            train_gen = train_datagen.flow_from_directory(
                TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=individual['batch_size'], class_mode='binary', subset='training'
            )
            val_gen = train_datagen.flow_from_directory(
                TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=individual['batch_size'], class_mode='binary', subset='validation'
            )
            
            if train_gen.samples == 0:
                print("   ❌ Error: No images found. Check TRAIN_DIR.")
                return 0.0

        except Exception as e:
            print(f"   ❌ Data Generator Error: {e}")
            return 0.0

        # 2. Train
        try:
            model = build_model(individual)
            es = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
            
            # Using verbose=2 to reduce log spam (one line per epoch)
            history = model.fit(
                train_gen,
                epochs=EPOCHS_PER_EVAL,
                validation_data=val_gen,
                steps_per_epoch=int(max(1, train_gen.samples // individual['batch_size'])),
                validation_steps=int(max(1, val_gen.samples // individual['batch_size'])),
                callbacks=[es],
                verbose=2
            )
            
            best_acc = max(history.history['val_accuracy'])
            print(f"   ✅ Acc: {best_acc:.4f}")
            
            tf.keras.backend.clear_session()
            del model
            return best_acc
            
        except Exception as e:
            print(f"   ❌ Training Failed: {e}")
            # Use traceback to see exactly where logic fails if another error occurs
            traceback.print_exc() 
            return 0.0

    def mutate(self, individual, mutation_rate=0.2):
        for key in self.search_space.keys():
            if random.random() < mutation_rate:
                individual[key] = random.choice(self.search_space[key])
        return individual

    def crossover(self, parent1, parent2):
        child = {}
        for key in self.search_space.keys():
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child

    def run(self):
        print("="*70)
        print(f"🚀 GENETIC ALGORITHM SIMULATION")
        print(f"   Population: {self.pop_size} | Generations: {self.generations}")
        print("="*70)
        
        population = self.initialize_population()
        
        for gen in range(self.generations):
            print(f"\n--- GENERATION {gen + 1} / {self.generations} ---")
            fitness_scores = []
            
            for i, ind in enumerate(population):
                score = self.evaluate_fitness(ind, gen, i)
                fitness_scores.append((ind, score))
                
                # Log history
                log_entry = ind.copy()
                log_entry['generation'] = gen + 1
                log_entry['accuracy'] = score
                self.history.append(log_entry)
            
            # Sort by Fitness (Descending)
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            best_ind, best_score = fitness_scores[0]
            
            print(f"🏆 Gen {gen+1} Winner: {best_score:.4%} [{best_ind['architecture']}]")
            
            # Elitism: Top 40% survive
            survivor_count = int(self.pop_size * 0.4)
            survivors = [x[0] for x in fitness_scores[:survivor_count]]
            
            # Reproduction
            new_population = survivors[:] 
            while len(new_population) < self.pop_size:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
                
            population = new_population

        # Save Results
        df = pd.DataFrame(self.history)
        df = df.sort_values(by='accuracy', ascending=False)
        csv_path = 'ga_simulation_results.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\n🎉 OPTIMIZATION COMPLETE. Results: {csv_path}")
        if not df.empty:
            best = df.iloc[0]
            print("\n💡 RECOMMENDATION:")
            print(f"   Architecture: {best['architecture']}")
            print(f"   Optimizer:    {best['optimizer']}")
            print(f"   LR:           {best['learning_rate']}")
            print(f"   Dropout:      {best['dropout_rate']}")

if __name__ == '__main__':
    if not SIMULATION_MODE and not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: {TRAIN_DIR} not found. Please ensure augmented data exists.")
    else:
        ga = GeneticOptimizer(SEARCH_SPACE, POPULATION_SIZE, GENERATIONS)
        ga.run()