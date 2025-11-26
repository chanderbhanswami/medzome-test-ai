import os
import sys
import cv2
import numpy as np
import json
import time
import tensorflow as tf 

# --- IMPORT QUANTIFIER ---
try:
    from lfa_quantifier import LFAQuantifier
except ImportError:
    print("❌ Error: lfa_quantifier.py not found. Please ensure it is in the src/ folder.")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = '../models/medzome_final_model.keras'
IMG_HEIGHT = 384
IMG_WIDTH = 128
TEMP_CROP_PATH = 'temp_autocrop.jpg'

class MedzomeSystem:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        print(f"📦 Initializing Medzome System...")
        
        # 1. Initialize the Quantifier (The "Eye")
        self.quantifier = LFAQuantifier()
        
        # 2. Initialize the AI (The "Brain")
        self._load_model()
        print("✅ System Ready")

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"❌ Failed to load Keras model: {e}")
            raise

    def smart_crop(self, img):
        """
        Robust cropping to find the strip.
        Matches Training Logic (Full Width) to ensure AI accuracy.
        """
        # 1. Find the Anchor (Black ColorChecker Card)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img, False # No anchor found

        # Find largest dark object (The Card)
        card_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(card_contour) < 5000: 
            return img, False 

        x, y, w, h = cv2.boundingRect(card_contour)

        # 2. Define Search Zone (Strictly Right of Card)
        h_img, w_img = img.shape[:2]
        zone_x = x + w + 20
        zone_y = max(0, y - 50)
        zone_h = h + 100
        
        if zone_x >= w_img: return img, False

        search_zone = img[zone_y:min(h_img, zone_y+zone_h), zone_x:min(w_img, zone_x+600)]
        
        if search_zone.size == 0: return img, False

        # 3. Find the White Strip inside the zone
        zone_gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(zone_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 4)
        
        strip_cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_crop = None
        max_area = 0
        
        for cnt in strip_cnts:
            area = cv2.contourArea(cnt)
            if area > 3000: # Must be substantial size
                sx, sy, sw, sh = cv2.boundingRect(cnt)
                aspect = float(sh)/sw
                
                # LFA Strip Physics: Tall and Thin (Ratio 2.0 to 6.0)
                if 2.0 < aspect < 6.0:
                    if area > max_area:
                        max_area = area
                        
                        # --- FIX: Capture Full Width (Matches Training Data) ---
                        # AI needs to see the plastic edges to orient itself.
                        # The Quantifier will handle inner cropping internally.
                        best_crop = search_zone[sy:sy+sh, sx:sx+sw]
        
        if best_crop is not None:
            return best_crop, True
                    
        return img, False

    def preprocess_for_ai(self, img_path):
        img = cv2.imread(img_path)
        if img is None: raise ValueError("Failed to read image")

        # Resize
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        # RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize (-1 to 1 for MobileNetV2)
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        # Batch dim
        return np.expand_dims(img, axis=0)

    def analyze(self, image_path):
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}

        start_time = time.time()
        
        # --- STEP 0: SMART CROP (Auto-Detection) ---
        original_img = cv2.imread(image_path)
        cropped_img, was_cropped = self.smart_crop(original_img)
        
        analysis_path = image_path
        
        # If we cropped a raw photo, save it temporarily
        if was_cropped:
            cv2.imwrite(TEMP_CROP_PATH, cropped_img)
            analysis_path = TEMP_CROP_PATH

        # --- STEP 1: QUANTITATIVE ANALYSIS ---
        q_metrics = self.quantifier.process_strip(analysis_path)
        test_intensity = q_metrics['test_intensity']
        
        # --- STEP 2: AI INFERENCE ---
        try:
            input_data = self.preprocess_for_ai(analysis_path)
            predictions = self.model.predict(input_data, verbose=0)
            ai_score = float(predictions[0][0])
        except Exception as e:
            return {"error": f"AI Inference failed: {str(e)}"}

        # --- STEP 3: HYBRID DECISION LOGIC ---
        final_status = "Negative"
        method = "AI_Agreement"
        
        # LOGIC GATES
        
        # 1. Strong Positive: AI is sure (>70%) AND Intensity is visible (>3.0)
        if ai_score > 0.7 and test_intensity > 3.0:
            final_status = "Positive"
            method = "Confirmed_Positive"
            
        # 2. Faint Positive: AI is VERY sure (>95%) AND we see a faint signal (>1.5)
        elif ai_score > 0.95 and test_intensity > 1.5:
            final_status = "Positive"
            method = "AI_Rescued_Faint_Line"
            
        # 3. Signal Override: AI missed it, but intensity is huge (>20.0)
        # Increased threshold to avoid false positives from shadows
        elif test_intensity > 20.0:
            final_status = "Positive"
            method = "Signal_Override"
            
        # 4. Negative / Veto
        else:
            final_status = "Negative"
            if ai_score > 0.5:
                method = "Quantifier_Veto" # Hallucination blocked
            else:
                method = "Confirmed_Negative"

        inference_time_ms = (time.time() - start_time) * 1000

        # Cleanup temp file
        if os.path.exists(TEMP_CROP_PATH):
            os.remove(TEMP_CROP_PATH)

        return {
            "file": os.path.basename(image_path),
            "input_type": "Raw Photo (Auto-Cropped)" if was_cropped else "Cropped Strip",
            "diagnosis": final_status,
            "decision_method": method,
            "confidence_score": f"{ai_score:.4f}",
            "quantitative_data": {
                "control_line": f"{q_metrics['control_intensity']:.2f}",
                "test_line": f"{q_metrics['test_intensity']:.2f}",
                "ratio_tc": f"{q_metrics.get('ratio', 0):.4f}"
            },
            "processing_time_ms": f"{inference_time_ms:.1f}"
        }

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        try:
            system = MedzomeSystem()
            print(f"\n🔎 Analyzing: {test_file}")
            result = system.analyze(test_file)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"❌ Critical Error: {e}")
    else:
        print("Usage: python final_inference.py /path/to/image.jpg")