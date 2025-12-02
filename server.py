#!/usr/bin/env python3
"""
Medzome Flask Backend Server
Supports TFLite, Keras (.keras), and HDF5 (.h5) models
Universal model serving with maximum capability utilization

Now includes:
- LFAQuantifier integration for accurate line intensity detection
- Smart crop functionality for auto-detection from raw photos
- Hybrid decision logic (AI + Quantitative analysis)
"""

import os
import sys
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, send_from_directory
from pathlib import Path
import time
from typing import Tuple, Dict, Any
import json

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not found")

# TFLite Runtime import
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME_AVAILABLE = True
    print("✅ TFLite runtime available")
except ImportError:
    TFLITE_RUNTIME_AVAILABLE = False
    print("⚠️  TFLite runtime not found")

# LFAQuantifier import (for accurate line intensity detection)
try:
    from lfa_quantifier import LFAQuantifier
    QUANTIFIER_AVAILABLE = True
    print("✅ LFAQuantifier available")
except ImportError:
    QUANTIFIER_AVAILABLE = False
    print("⚠️  LFAQuantifier not found - quantitative analysis disabled")

# SciPy for signal processing (used by LFAQuantifier)
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  SciPy not found - some analysis features may be limited")

# Intensity Normalizer import (for calibrated intensity readings)
try:
    from intensity_normalizer import IntensityNormalizer
    import pickle
    NORMALIZER_AVAILABLE = True
    print("✅ IntensityNormalizer available")
except ImportError:
    NORMALIZER_AVAILABLE = False
    IntensityNormalizer = None
    print("⚠️  IntensityNormalizer not found - normalization disabled")

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Changed default to Keras model for better accuracy with quantitative analysis
    MODEL_PATH = os.environ.get('MODEL_PATH', 'medzome_final_model.keras')
    INPUT_HEIGHT = 384
    INPUT_WIDTH = 128
    INPUT_CHANNELS = 3
    CONFIDENCE_THRESHOLD = 0.5
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Model type auto-detection
    MODEL_TYPE = None  # Will be auto-detected: 'tflite', 'keras', 'h5'
    
    # Model-specific configurations (can be customized per model)
    MODEL_CONFIGS = {
        'medzome_efficientnet.keras': {
            'threshold': 0.4,  # Lower threshold for EfficientNet
            'invert_prediction': False
        },
        'medzome_final_model.keras': {
            'threshold': 0.5,  # Optimal threshold for final model
            'invert_prediction': False,
            'use_quantifier': True  # Enable LFAQuantifier for this model
        },
        'default': {
            'threshold': 0.5,
            'invert_prediction': False
        }
    }
    
    # Hybrid decision thresholds (EXACT values from final_inference.py)
    # DO NOT CHANGE - These are calibrated for the trained model
    STRONG_POSITIVE_AI_THRESHOLD = 0.7
    STRONG_POSITIVE_INTENSITY_THRESHOLD = 15.0  # Matches final_inference.py
    FAINT_POSITIVE_AI_THRESHOLD = 0.998         # Matches final_inference.py
    FAINT_POSITIVE_INTENSITY_THRESHOLD = 1.5    # Matches final_inference.py
    SIGNAL_OVERRIDE_INTENSITY_THRESHOLD = 20.0  # Matches final_inference.py
    
    # Normalization model paths (relative to script location)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    NORMALIZER_PATH = os.path.join(SCRIPT_DIR, 'intensity_normalizer.pkl')
    NORMALIZATION_PARAMS_PATH = os.path.join(SCRIPT_DIR, 'normalization_params.json')

config = Config()

# ============================================================================
# Flask App Setup
# ============================================================================

app = Flask(__name__, static_folder='.', static_url_path='')
# CORS not needed - single service serves both frontend and API

# ============================================================================
# Universal Model Loader
# ============================================================================

class UniversalModelLoader:
    """
    Universal model loader that can handle TFLite, Keras, and H5 formats.
    Automatically detects model type and loads with maximum capability.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.model_type = None
        self.input_details = None
        self.output_details = None
        self.interpreter = None
        
        self._detect_and_load_model()
    
    def _detect_model_type(self) -> str:
        """Detect model type from file extension"""
        ext = Path(self.model_path).suffix.lower()
        
        if ext == '.tflite':
            return 'tflite'
        elif ext == '.keras':
            return 'keras'
        elif ext == '.h5':
            return 'h5'
        else:
            raise ValueError(f"Unsupported model format: {ext}")
    
    def _detect_and_load_model(self):
        """Detect model type and load appropriately"""
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model_type = self._detect_model_type()
        
        print(f"\n📦 Loading model: {self.model_path}")
        print(f"   Type: {self.model_type.upper()}")
        print(f"   Size: {os.path.getsize(self.model_path) / 1024:.1f} KB")
        
        if self.model_type == 'tflite':
            self._load_tflite()
        elif self.model_type == 'keras':
            self._load_keras()
        elif self.model_type == 'h5':
            self._load_h5()
        
        # Apply model-specific configuration
        model_name = os.path.basename(self.model_path)
        if model_name in config.MODEL_CONFIGS:
            model_config = config.MODEL_CONFIGS[model_name]
            config.CONFIDENCE_THRESHOLD = model_config['threshold']
            print(f"   ⚙️  Custom threshold: {config.CONFIDENCE_THRESHOLD}")
        
        print(f"   ✅ Model loaded successfully")
    
    def _load_tflite(self):
        """Load TFLite model"""
        try:
            # Try TFLite runtime first (faster)
            if TFLITE_RUNTIME_AVAILABLE:
                print("   Using: TFLite Runtime")
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            elif TF_AVAILABLE:
                print("   Using: TensorFlow Lite")
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            else:
                raise RuntimeError("No TFLite runtime available")
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model: {e}")
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get model's expected input shape (height, width, channels)"""
        if self.model_type == 'tflite':
            shape = self.input_details[0]['shape']
            # Shape format: [batch, height, width, channels]
            return int(shape[1]), int(shape[2]), int(shape[3])
        elif self.model_type in ['keras', 'h5']:
            shape = self.model.input_shape
            # Shape format: (batch, height, width, channels) or (None, height, width, channels)
            return int(shape[1]), int(shape[2]), int(shape[3])
        else:
            # Fallback to default
            return config.INPUT_HEIGHT, config.INPUT_WIDTH, config.INPUT_CHANNELS
    
    def _load_keras(self):
        """Load Keras (.keras) model"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required for Keras models")
        
        try:
            print("   Using: TensorFlow/Keras")
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Get input/output shapes
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")
    
    def _load_h5(self):
        """Load HDF5 (.h5) model"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required for H5 models")
        
        try:
            print("   Using: TensorFlow/Keras (H5)")
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Get input/output shapes
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load H5 model: {e}")
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on preprocessed image.
        Returns prediction with maximum capability based on model type.
        """
        
        if self.model_type == 'tflite':
            return self._predict_tflite(image)
        elif self.model_type in ['keras', 'h5']:
            return self._predict_keras(image)
    
    def _predict_tflite(self, image: np.ndarray) -> Dict[str, Any]:
        """TFLite inference"""
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Debug: Print output shape and values
        print(f"   Model output shape: {output.shape}")
        print(f"   Model output values: {output[0]}")
        
        # Extract confidence based on output format
        if output.shape[-1] == 1:
            # Single output (sigmoid activation)
            confidence = float(output[0][0])
        elif output.shape[-1] == 2:
            # Binary classification (softmax with 2 classes)
            confidence = float(output[0][1])
            print(f"   Binary classification: negative={output[0][0]:.4f}, positive={output[0][1]:.4f}")
        else:
            # Multi-class or other format - use max probability
            confidence = float(np.max(output[0]))
            print(f"   Using max probability from {output.shape[-1]} classes")
        
        return {
            'confidence': confidence,
            'raw_output': output.tolist()
        }
    
    def _predict_keras(self, image: np.ndarray) -> Dict[str, Any]:
        """Keras/H5 inference"""
        # Run prediction
        output = self.model.predict(image, verbose=0)
        
        # Debug: Print output shape and values
        print(f"   Model output shape: {output.shape}")
        print(f"   Model output values: {output[0]}")
        
        # Extract confidence based on output format
        if output.shape[-1] == 1:
            # Single output (sigmoid activation)
            confidence = float(output[0][0])
        elif output.shape[-1] == 2:
            # Binary classification (softmax with 2 classes)
            # Class 0: negative, Class 1: positive
            confidence = float(output[0][1])
            print(f"   Binary classification: negative={output[0][0]:.4f}, positive={output[0][1]:.4f}")
        else:
            # Multi-class or other format - use max probability
            confidence = float(np.max(output[0]))
            print(f"   Using max probability from {output.shape[-1]} classes")
        
        return {
            'confidence': confidence,
            'raw_output': output.tolist()
        }

# ============================================================================
# Smart Crop Processor (from final_inference.py)
# ============================================================================


class SmartCropProcessor:
    """
    Smart cropping to find and isolate test strips from raw photos.
    Matches Training Logic to ensure AI accuracy.
    """
    
    @staticmethod
    def smart_crop(img):
        """
        Robust cropping to find the strip from raw photos.
        Matches Training Logic (Full Width) to ensure AI accuracy.
        """
        try:
            import cv2
            import numpy as np
            
            # 1. Find the Anchor (Black ColorChecker Card)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return img, False  # No anchor found
            
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
            
            if zone_x >= w_img:
                return img, False
            
            search_zone = img[zone_y:min(h_img, zone_y + zone_h), zone_x:min(w_img, zone_x + 1200)]
            
            if search_zone.size == 0:
                return img, False
            
            # 3. Find the White Strip inside the zone using Edge Detection
            # (More robust for white-on-white than thresholding)
            zone_gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(zone_gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # Dilate to connect edges
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            strip_cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_crop = None
            max_area = 0
            
            for cnt in strip_cnts:
                area = cv2.contourArea(cnt)
                if area > 3000:  # Must be substantial size
                    sx, sy, sw, sh = cv2.boundingRect(cnt)
                    aspect = float(sh) / sw if sw > 0 else 0
                    
                    # LFA Strip Physics: Tall and Thin (Ratio 2.0 to 6.0)
                    if 2.0 < aspect < 6.0:
                        if area > max_area:
                            max_area = area
                            # Capture Full Width (Matches Training Data)
                            best_crop = search_zone[sy:sy + sh, sx:sx + sw]
            
            if best_crop is not None:
                return best_crop, True

            # --- FALLBACK: Direct Cassette Detection (No Card) ---
            try:
                scale = 0.5
                small_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
                hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
                lower_white = np.array([0, 0, 180])
                upper_white = np.array([180, 50, 255])
                mask = cv2.inRange(hsv, lower_white, upper_white)
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_rect = None
                max_area = 0
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < (2000 * scale * scale): continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = float(h)/w
                    if 1.5 < aspect < 5.0:
                        if area > max_area:
                            max_area = area
                            best_rect = (x, y, w, h)
                
                if best_rect:
                    x, y, w, h = best_rect
                    x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    pad_x = int(w * 0.1)
                    pad_y = int(h * 0.05)
                    h_img, w_img = img.shape[:2]
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w_img, x + w + pad_x)
                    y2 = min(h_img, y + h + pad_y)
                    return img[y1:y2, x1:x2], True
            except Exception as e:
                print(f"⚠️  Fallback crop failed: {e}")

            # --- FALLBACK 2: Strip Extraction (Assume Image IS Cassette) ---
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 15, -5)
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_strip = None
                best_score = 0
                h, w = img.shape[:2]
                center_x = w // 2
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 500: continue
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect = float(ch) / cw if cw > 0 else 0
                    if aspect > 2.0:
                        contour_center = x + cw // 2
                        distance_from_center = abs(contour_center - center_x)
                        center_score = 1.0 / (1.0 + distance_from_center / w)
                        score = area * center_score
                        if score > best_score:
                            best_score = score
                            best_strip = (x, y, cw, ch)
                
                if best_strip:
                    x, y, cw, ch = best_strip
                    pad_x = int(cw * 0.1)
                    pad_y = int(ch * 0.02)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w, x + cw + pad_x)
                    y2 = min(h, y + ch + pad_y)
                    return img[y1:y2, x1:x2], True
            except Exception as e:
                print(f"⚠️  Strip fallback failed: {e}")
            
            return img, False
            
        except Exception as e:
            print(f"⚠️  Smart crop failed: {e}")
            return img, False
    
    @staticmethod
    def decode_base64_to_cv2(image_data: str):
        """Decode base64 image to OpenCV BGR format"""
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


# ============================================================================
# LFA Quantifier Wrapper
# ============================================================================

class LFAQuantifierWrapper:
    """
    Wrapper for LFAQuantifier to work with in-memory images.
    Provides quantitative analysis of test strips.
    """
    
    def __init__(self):
        if QUANTIFIER_AVAILABLE:
            self.quantifier = LFAQuantifier()
        else:
            self.quantifier = None
    
    def process_strip_from_array(self, img):
        """
        Process strip image directly from numpy array.
        """
        if img is None:
            return {"error": "Image not found", "result": "Error", 
                    "control_intensity": 0.0, "test_intensity": 0.0, "ratio": 0.0}
        
        try:
            h, w = img.shape[:2]
            
            # 1. Wider Crop (30% to 70%)
            center_crop = img[10:h-10, int(w*0.30):int(w*0.70)]
            
            # 2. Invert Green Channel
            b, g, r = cv2.split(center_crop)
            signal = 255 - g
            
            # 3. Create Profile
            profile = np.mean(signal, axis=1)
            
            # 4. Find Control Line
            top_half_profile = profile[:int(h*0.6)]
            
            results = {
                "control_intensity": 0.0,
                "test_intensity": 0.0,
                "ratio": 0.0,
                "result": "Invalid"
            }
            
            # Find Control Peak
            if SCIPY_AVAILABLE:
                c_peaks, _ = find_peaks(top_half_profile, height=20, distance=20)
            else:
                c_peaks = [] # Fallback omitted for brevity
            
            if len(c_peaks) > 0:
                c_pos = c_peaks[np.argmax(top_half_profile[c_peaks])]
                c_intensity = float(profile[c_pos])
                results['control_intensity'] = c_intensity
                
                # 5. Targeted Test Line Search
                start_search = c_pos + 60
                end_search = min(c_pos + 160, h-10)
                
                if start_search < end_search:
                    test_zone = profile[start_search:end_search]
                    max_signal_idx = np.argmax(test_zone)
                    max_signal = test_zone[max_signal_idx]
                    background = np.min(test_zone)
                    true_intensity = float(max_signal - background)
                    
                    results['test_intensity'] = max(0.0, true_intensity)
                    
                    if results['test_intensity'] > 3.0:
                        results['result'] = "Positive"
                    else:
                        results['result'] = "Negative"
                    
                    if c_intensity > 0:
                        results['ratio'] = results['test_intensity'] / c_intensity
            
            return results
            
        except Exception as e:
            print(f"⚠️  Quantifier analysis failed: {e}")
            return {
                "control_intensity": 0.0,
                "test_intensity": 0.0,
                "ratio": 0.0,
                "result": "Error",
                "error": str(e)
            }


# ============================================================================
# Hybrid Decision Engine (from final_inference.py)
# ============================================================================

class HybridDecisionEngine:
    """
    Hybrid decision logic combining AI confidence with quantitative analysis.
    EXACTLY matches the logic in final_inference.py for 100% identical results.
    """
    
    @staticmethod
    def make_decision(ai_score, quantitative_data):
        test_intensity = quantitative_data.get('test_intensity', 0.0)
        
        final_status = "Negative"
        method = "AI_Agreement"
        
        # 1. Strong Positive
        if ai_score > 0.7 and test_intensity > 15.0:
            final_status = "Positive"
            method = "Confirmed_Positive"
            
        # 2. Faint Positive
        elif ai_score > 0.998 and test_intensity > 1.5:
            final_status = "Positive"
            method = "AI_Rescued_Faint_Line"
            
        # 3. Signal Override
        elif test_intensity > 20.0:
            final_status = "Positive"
            method = "Signal_Override"
            
        # 4. Negative / Veto
        else:
            final_status = "Negative"
            if ai_score > 0.5:
                method = "Quantifier_Veto"
            else:
                method = "Confirmed_Negative"
        
        return {
            "diagnosis": final_status,
            "decision_method": method
        }


# ============================================================================
# Intensity Normalizer Wrapper
# ============================================================================

class IntensityNormalizerWrapper:
    def __init__(self):
        self.normalizer = None
        self.norm_params = None
        self._load_normalizer()
    
    def _load_normalizer(self):
        # Try to load pickle model
        if NORMALIZER_AVAILABLE and os.path.exists(config.NORMALIZER_PATH):
            try:
                with open(config.NORMALIZER_PATH, 'rb') as f:
                    self.normalizer = pickle.load(f)
            except Exception as e:
                print(f"⚠️  Could not load normalizer pickle: {e}")
        
        # Load JSON params
        if os.path.exists(config.NORMALIZATION_PARAMS_PATH):
            try:
                with open(config.NORMALIZATION_PARAMS_PATH, 'r') as f:
                    self.norm_params = json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load normalization params: {e}")
    
    def normalize(self, test_intensity, control_intensity, ai_prediction, ai_confidence):
        result = {
            'normalized_intensity': test_intensity,
            'estimated_concentration': None,
            'normalization_applied': False,
            'method': 'none'
        }
        
        if self.normalizer is not None:
            try:
                norm_result = self.normalizer.normalize(
                    test_intensity=test_intensity,
                    control_intensity=control_intensity,
                    ai_prediction=ai_prediction,
                    ai_confidence=ai_confidence
                )
                result['normalized_intensity'] = norm_result['normalized_intensity']
                result['estimated_concentration'] = norm_result.get('estimated_concentration')
                result['normalization_applied'] = True
                result['method'] = 'full_model'
                return result
            except Exception as e:
                print(f"⚠️  Normalizer error: {e}, using fallback")
        
        if self.norm_params is not None:
            try:
                threshold = self.norm_params['thresholds']['negative_intensity_threshold']
                baseline = self.norm_params['thresholds']['baseline']
                
                if ai_prediction.lower() == 'negative':
                    result['normalized_intensity'] = 0.0
                    result['estimated_concentration'] = 0.0
                    result['normalization_applied'] = True
                    result['method'] = 'threshold_negative'
                else:
                    normalized = test_intensity - baseline
                    normalized = max(0, normalized)
                    scale = self.norm_params.get('normalization_scale', 1.0)
                    if scale > 0:
                        normalized = (normalized / scale) * 100
                    result['normalized_intensity'] = round(normalized, 2)
                    result['normalization_applied'] = True
                    result['method'] = 'threshold_positive'
                return result
            except Exception as e:
                print(f"⚠️  Params fallback error: {e}")
        
        if ai_prediction.lower() == 'negative':
            result['normalized_intensity'] = 0.0
            result['method'] = 'simple_threshold'
        
        return result

# Global normalizer instance
intensity_normalizer = None

def load_normalizer():
    global intensity_normalizer
    try:
        intensity_normalizer = IntensityNormalizerWrapper()
    except Exception as e:
        print(f"⚠️  Failed to initialize normalizer: {e}")
        intensity_normalizer = None


# ============================================================================
# API Endpoints
# ============================================================================

# Global model instance
model_loader = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    if path in ['predict', 'health', 'analyze'] or path.startswith('model/'):
        return None
    return send_from_directory('.', path)

def load_model():
    global model_loader, intensity_normalizer
    try:
        model_loader = UniversalModelLoader(config.MODEL_PATH)
        print("✅ AI model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    try:
        intensity_normalizer = IntensityNormalizerWrapper()
        print("✅ Intensity normalizer loaded successfully")
    except Exception as e:
        intensity_normalizer = None
    
    print("✅ Server ready to serve predictions")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader is not None,
        'model_type': model_loader.model_type if model_loader else None,
        'model_path': config.MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint with hybrid decision logic.
    """
    TEMP_CROP_PATH = 'temp_autocrop.jpg'
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = data['image']
        threshold = data.get('threshold', config.CONFIDENCE_THRESHOLD)
        config.CONFIDENCE_THRESHOLD = threshold
        
        start_time = time.time()
        
        # Decode original image
        original_image = SmartCropProcessor.decode_base64_to_cv2(image_data)
        
        if original_image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # --- STEP 0: SMART CROP (Auto-Detection) ---
        cropped_image, was_cropped = SmartCropProcessor.smart_crop(original_image)
        input_type = "Raw Photo (Auto-Cropped)" if was_cropped else "Cropped Strip"
        
        print(f"   Original image: {original_image.shape}")
        print(f"   Was cropped: {was_cropped}")
        if was_cropped:
            print(f"   Cropped image: {cropped_image.shape}")
        
        if was_cropped:
            # Resize to standard size (128x384) for consistent analysis
            cropped_image = cv2.resize(cropped_image, (config.INPUT_WIDTH, config.INPUT_HEIGHT))
            # Save cropped image to JPEG temp file (like final_inference.py)
            cv2.imwrite(TEMP_CROP_PATH, cropped_image)
            # Read back for quantifier (JPEG compression applied)
            analysis_image = cv2.imread(TEMP_CROP_PATH)
            used_temp_file = True
        else:
            # Use original image directly (no JPEG re-encoding)
            analysis_image = original_image
            used_temp_file = False
        # --- STEP 1: QUANTITATIVE ANALYSIS ---
        # Process in-memory to match what final_inference.py does after cv2.imread
        h, w = analysis_image.shape[:2]
        
        # Wider Crop (30% to 70%) to ensure we don't miss off-center lines
        center_crop = analysis_image[10:h-10, int(w*0.30):int(w*0.70)]
        
        # Invert Green Channel (Lines become bright)
        b, g, r = cv2.split(center_crop)
        signal = 255 - g
        
        # Create Profile
        profile = np.mean(signal, axis=1)
        
        # Find Control Line (The strongest peak in top half)
        # IMPORTANT: Use original h (not profile length) to match lfa_quantifier.py
        top_half_profile = profile[:int(h*0.6)]
        
        quantitative_data = {
            "control_intensity": 0.0,
            "test_intensity": 0.0,
            "ratio": 0.0,
            "result": "Invalid"
        }
        
        # Find Control Peak using scipy
        if SCIPY_AVAILABLE:
            c_peaks, _ = find_peaks(top_half_profile, height=20, distance=20)
        else:
            # Fallback: simple peak detection
            c_peaks = []
            for i in range(20, len(top_half_profile) - 20):
                if top_half_profile[i] > 20:
                    is_peak = True
                    for j in range(1, 21):
                        if top_half_profile[i] <= top_half_profile[i - j] or top_half_profile[i] <= top_half_profile[i + j]:
                            is_peak = False
                            break
                    if is_peak:
                        c_peaks.append(i)
            c_peaks = np.array(c_peaks)
        
        if len(c_peaks) > 0:
            # Pick the tallest peak as Control
            c_pos = c_peaks[np.argmax(top_half_profile[c_peaks])]
            c_intensity = float(profile[c_pos])
            quantitative_data['control_intensity'] = c_intensity
            
            # Targeted Test Line Search
            start_search = c_pos + 60
            end_search = min(c_pos + 160, h-10)
            
            if start_search < end_search:
                test_zone = profile[start_search:end_search]
                
                # Look for MAXIMUM signal in this zone
                max_signal_idx = np.argmax(test_zone)
                max_signal = test_zone[max_signal_idx]
                
                # Calculate local background
                background = np.min(test_zone)
                true_intensity = float(max_signal - background)
                
                quantitative_data['test_intensity'] = max(0.0, true_intensity)
                
                # Decision Logic
                if quantitative_data['test_intensity'] > 3.0:
                    quantitative_data['result'] = "Positive"
                else:
                    quantitative_data['result'] = "Negative"
                
                if c_intensity > 0:
                    quantitative_data['ratio'] = quantitative_data['test_intensity'] / c_intensity
        
        print(f"   Quantitative Analysis: control={quantitative_data['control_intensity']:.2f}, test={quantitative_data['test_intensity']:.2f}")
        
        # --- STEP 2: AI INFERENCE ---
        # Get model's required input dimensions
        height, width, channels = model_loader.get_input_shape()
        
        # Preprocess EXACTLY like final_inference.py preprocess_for_ai()
        # Uses analysis_image which is either JPEG-decoded (if cropped) or original (if not)
        img_for_ai = cv2.resize(analysis_image, (width, height))
        img_for_ai = cv2.cvtColor(img_for_ai, cv2.COLOR_BGR2RGB)
        img_for_ai = img_for_ai.astype(np.float32)
        img_for_ai = (img_for_ai / 127.5) - 1.0
        processed_image = np.expand_dims(img_for_ai, axis=0)
        
        # Run inference
        prediction = model_loader.predict(processed_image)
        ai_score = prediction['confidence']
        
        print(f"   AI Score: {ai_score:.4f}")
        
        # --- STEP 3: HYBRID DECISION LOGIC ---
        decision = HybridDecisionEngine.make_decision(ai_score, quantitative_data)
        diagnosis = decision['diagnosis']
        decision_method = decision['decision_method']
        
        print(f"   Decision: {diagnosis} (method: {decision_method})")
        
        # --- STEP 4: INTENSITY NORMALIZATION ---
        # Apply normalization to get calibrated intensity values (matches final_inference.py)
        normalized_data = {
            'normalized_intensity': quantitative_data['test_intensity'],
            'estimated_concentration': None,
            'normalization_applied': False,
            'method': 'none'
        }
        
        if intensity_normalizer is not None:
            normalized_data = intensity_normalizer.normalize(
                test_intensity=quantitative_data['test_intensity'],
                control_intensity=quantitative_data['control_intensity'],
                ai_prediction=diagnosis,
                ai_confidence=ai_score
            )
            print(f"   Normalization: raw={quantitative_data['test_intensity']:.2f} -> normalized={normalized_data['normalized_intensity']:.2f}, method={normalized_data['method']}")
        else:
            print("   ⚠️  Normalizer not loaded - using raw intensity values")
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Cleanup temp file if we used it
        if used_temp_file and os.path.exists(TEMP_CROP_PATH):
            os.remove(TEMP_CROP_PATH)
        
        # Build response matching final_inference.py output EXACTLY
        response = {
            'success': True,
            
            # Primary result (matches final_inference.py exactly)
            'diagnosis': diagnosis,
            'is_positive': diagnosis == "Positive",
            'decision_method': decision_method,
            'input_type': input_type,
            
            # AI confidence
            'confidence': ai_score,
            'confidence_score': f"{ai_score:.4f}",
            
            # Quantitative data (matching final_inference.py format exactly)
            'quantitative_data': {
                'control_line': f"{quantitative_data['control_intensity']:.2f}",
                'test_line': f"{quantitative_data['test_intensity']:.2f}",  # Backward compatibility
                'test_line_raw': f"{quantitative_data['test_intensity']:.2f}",
                'test_line_normalized': f"{normalized_data['normalized_intensity']:.2f}",
                'ratio_tc': f"{quantitative_data.get('ratio', 0):.4f}",
                'estimated_concentration_ng_ml': normalized_data['estimated_concentration']
            },
            
            # Normalization info
            'normalization': {
                'applied': normalized_data['normalization_applied'],
                'method': normalized_data['method']
            },
            
            # Backward compatibility fields
            'control_line_detected': quantitative_data['control_intensity'] > 0,
            'test_line_detected': quantitative_data['test_intensity'] > config.STRONG_POSITIVE_INTENSITY_THRESHOLD,
            
            # Processing info
            'processing_time_ms': processing_time,
            'model_type': model_loader.model_type,
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        # Cleanup temp file on error
        if os.path.exists('temp_autocrop.jpg'):
            os.remove('temp_autocrop.jpg')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    
    if not model_loader:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get actual input shape from loaded model
    height, width, channels = model_loader.get_input_shape()
    
    info = {
        'model_path': config.MODEL_PATH,
        'model_type': model_loader.model_type,
        'model_size_kb': os.path.getsize(config.MODEL_PATH) / 1024,
        'input_shape': [1, height, width, channels],
        'input_height': height,
        'input_width': width,
        'input_channels': channels,
        'confidence_threshold': config.CONFIDENCE_THRESHOLD
    }
    
    return jsonify(info)

@app.route('/model/change', methods=['POST'])
def change_model():
    """
    Change the active model dynamically.
    Supports TFLite, Keras, and H5 formats.
    """
    global model_loader
    
    try:
        data = request.get_json()
        
        if not data or 'model_path' not in data:
            return jsonify({'error': 'No model_path provided'}), 400
        
        new_model_path = data['model_path']
        
        if not os.path.exists(new_model_path):
            return jsonify({'error': f'Model not found: {new_model_path}'}), 404
        
        # Load new model
        config.MODEL_PATH = new_model_path
        model_loader = UniversalModelLoader(new_model_path)
        
        return jsonify({
            'success': True,
            'message': 'Model changed successfully',
            'model_path': new_model_path,
            'model_type': model_loader.model_type
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# Load Model on Import (for Gunicorn/WSGI servers like Render)
# ============================================================================

load_model()

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Medzome Flask Backend Server')
    parser.add_argument('--model', default=config.MODEL_PATH, help='Model path')
    parser.add_argument('--threshold', type=float, help='Custom confidence threshold (0.0-1.0)')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=None, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    # Update config
    config.MODEL_PATH = args.model
    
    # Override threshold if provided via command line
    if args.threshold is not None:
        config.CONFIDENCE_THRESHOLD = args.threshold
        print(f"\n⚙️  Custom threshold from CLI: {config.CONFIDENCE_THRESHOLD}\n")
    
    # Get port from environment variable (Render sets this) or use default
    port = args.port or int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*70)
    print("MEDZOME FLASK BACKEND SERVER")
    print("="*70)
    print(f"Model: {config.MODEL_PATH}")
    print(f"Threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"Host: {args.host}")
    print(f"Port: {port}")
    print("="*70 + "\n")
    
    # Load model before starting server
    load_model()
    
    # Run server
    app.run(
        host=args.host,
        port=port,
        debug=args.debug,
        threaded=True
    )