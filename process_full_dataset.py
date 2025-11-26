import cv2
import numpy as np
import os
import shutil

# --- CONFIGURATION ---
RAW_DATA_ROOT = '../raw_data'       # Where your dilution folders are
OUTPUT_DIR = '../processed_data'    # Where standardized images go

# Detection Settings
SEARCH_OFFSET_X = 20    # Start looking 20px right of the black card
SEARCH_WIDTH = 600      # Scan 600px wide for the strip
TARGET_SIZE = (128, 384) # Standard input size for AI & Quantifier

def smart_crop_and_save(image_path, save_path):
    """
    Finds the black card, locates the strip to its right, 
    centers on the strip, and saves a standardized 128x384 crop.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    # 1. Downscale slightly for faster detection (processing at full 12MP is slow)
    # We will scale coordinates back up later if needed, but for simplicity
    # we can process at native resolution if speed isn't critical.
    # Let's process native to keep maximum detail for faint lines.
    
    # 2. Find the Anchor (The Black ColorChecker Card)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find black objects (inverse binary)
    # Pixels darker than 80 become 255 (white), others 0 (black)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False

    # The card is usually the largest dark object
    card_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(card_contour) < 5000: # Ignore small specs
        return False

    x, y, w, h = cv2.boundingRect(card_contour)

    # 3. Define the Search Zone (Right of the card)
    zone_x = x + w + SEARCH_OFFSET_X
    zone_y = max(0, y - 50) # Look slightly above card top
    zone_h = h + 100        # Look slightly below card bottom
    
    img_h, img_w = img.shape[:2]
    
    # Ensure we don't go out of bounds
    safe_x2 = min(img_w, zone_x + SEARCH_WIDTH)
    safe_y2 = min(img_h, zone_y + zone_h)
    
    # Crop the Search Zone
    search_zone = img[zone_y:safe_y2, zone_x:safe_x2]
    
    if search_zone.size == 0:
        return False

    # 4. Smart Centering: Find the white strip inside the zone
    # We use adaptive thresholding to separate the white strip from the white background
    zone_gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresh is great for finding edges of the plastic casing
    thresh_strip = cv2.adaptiveThreshold(zone_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 4)

    strip_cnts, _ = cv2.findContours(thresh_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the best candidate for the strip
    # Criteria: Vertical rectangle, reasonable size
    best_crop = None
    
    # Sort by area large to small
    sorted_cnts = sorted(strip_cnts, key=cv2.contourArea, reverse=True)
    
    for cnt in sorted_cnts:
        area = cv2.contourArea(cnt)
        if area > 3000: # Strip must be substantial
            sx, sy, sw, sh = cv2.boundingRect(cnt)
            aspect = float(sh) / sw
            
            # LFA strips are tall and narrow (Aspect ratio ~2.0 to 5.0)
            if 1.5 < aspect < 6.0:
                # We found it!
                # Calculate center of the strip
                center_x = sx + (sw // 2)
                center_y = sy + (sh // 2)
                
                # Create a standardized crop box around this center
                # We want 128 width relative to 384 height
                # Let's crop tight to the detected height
                crop_h = sh
                crop_w = int(sh / 3) # Maintain 1:3 aspect ratio roughly
                
                # Coordinates in the zone
                start_x = max(0, center_x - (crop_w // 2))
                end_x = min(search_zone.shape[1], center_x + (crop_w // 2))
                start_y = sy
                end_y = sy + sh
                
                best_crop = search_zone[start_y:end_y, start_x:end_x]
                break
    
    # Fallback: If we couldn't isolate the strip contour, take the middle of the search zone
    if best_crop is None:
        mid_x = search_zone.shape[1] // 3 # Assuming strip is near the left of search zone
        mid_w = 200
        best_crop = search_zone[:, max(0, mid_x):min(search_zone.shape[1], mid_x+mid_w)]

    # 5. Final Resize
    try:
        final_img = cv2.resize(best_crop, TARGET_SIZE)
        cv2.imwrite(save_path, final_img)
        return True
    except Exception as e:
        return False

def main():
    print("="*60)
    print(" LFA DATASET PROCESSOR (Smart Centering)")
    print("="*60)
    
    # Setup Output Dirs
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning old data in {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
        
    pos_dir = os.path.join(OUTPUT_DIR, 'positive')
    neg_dir = os.path.join(OUTPUT_DIR, 'negative')
    os.makedirs(pos_dir)
    os.makedirs(neg_dir)
    
    total_count = 0
    success_count = 0
    
    # Walk raw data
    for root, dirs, files in os.walk(RAW_DATA_ROOT):
        folder_name = os.path.basename(root).lower()
        
        # Skip root folder
        if root == RAW_DATA_ROOT:
            continue
            
        # Determine Label
        is_positive = True
        if "0_0_ng" in folder_name or "control" in folder_name:
            is_positive = False
        elif "ng" in folder_name:
            is_positive = True
        else:
            continue # Skip unknown folders
            
        print(f"Processing: {folder_name} -> {'POSITIVE' if is_positive else 'NEGATIVE'}")
        
        save_folder = pos_dir if is_positive else neg_dir
        
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(save_folder, file)
                
                if smart_crop_and_save(src_path, dst_path):
                    success_count += 1
                total_count += 1
                
    print("-" * 60)
    print(f"Processing Complete.")
    print(f"Total Images Found: {total_count}")
    print(f"Successfully Cropped: {success_count}")
    print(f"Success Rate: {(success_count/total_count)*100:.1f}%")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()