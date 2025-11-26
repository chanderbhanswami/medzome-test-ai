import os
import numpy as np
import math
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# --- CONFIGURATION ---
INPUT_DIR = '../processed_data'
OUTPUT_DIR = '../augmented_data'
TARGET_PER_CLASS = 3000  # Aim for this many images per class to balance them

# Define the augmentation engine based on paper parameters
datagen = ImageDataGenerator(
    rotation_range=20,       
    zoom_range=0.1,          
    brightness_range=[0.9, 1.1], 
    width_shift_range=0.05,  
    height_shift_range=0.05, 
    fill_mode='nearest'      
)

def get_file_count(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def augment_dataset():
    # 1. Clean Output Directory to prevent mixing old data
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    classes = ['positive', 'negative']
    
    for class_name in classes:
        class_input_dir = os.path.join(INPUT_DIR, class_name)
        class_output_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Count original images
        num_files = get_file_count(class_input_dir)
        if num_files == 0:
            print(f"[Warning] No images found for class '{class_name}'")
            continue
            
        # Calculate how many augmentations needed per image to reach target
        # e.g., If we have 263 negatives and want 3000 -> Multiplier = ceil(3000/263) = 12
        augmentations_per_image = math.ceil(TARGET_PER_CLASS / num_files)
        
        print(f"Processing Class: {class_name}")
        print(f"  - Original Images: {num_files}")
        print(f"  - Augmentation Factor: {augmentations_per_image}x (Target: ~{TARGET_PER_CLASS})")
        
        processed_count = 0
        
        files = [f for f in os.listdir(class_input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in files:
            img_path = os.path.join(class_input_dir, filename)
            
            try:
                # Load Image
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape) 

                # 1. Save the Original Image first (Don't lose real data!)
                save_img(os.path.join(class_output_dir, f"orig_{filename}"), x[0])
                
                # 2. Generate Augmentations
                count = 0
                # Use a generous loop count, break when we hit the per-image target
                for batch in datagen.flow(x, batch_size=1):
                    save_name = f"aug_{count}_{filename}"
                    save_path = os.path.join(class_output_dir, save_name)
                    save_img(save_path, batch[0])
                    
                    count += 1
                    # Generate 'augmentations_per_image - 1' because we already saved the original
                    if count >= (augmentations_per_image - 1):
                        break 
                
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"    Processed {processed_count}/{num_files} images...")

            except Exception as e:
                print(f"[Error] Failed {filename}: {e}")

# --- RUN ---
if __name__ == "__main__":
    print("Starting Smart Balancing Augmentation...")
    augment_dataset()
    print(f"\nDone! Check '{OUTPUT_DIR}'")
    print("You should now have roughly equal numbers of Positive and Negative images.")