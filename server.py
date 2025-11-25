#!/usr/bin/env python3
"""
Medzome Flask Backend Server
Supports TFLite, Keras (.keras), and HDF5 (.h5) models
Universal model serving with maximum capability utilization
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

# ============================================================================
# Configuration
# ============================================================================

class Config:
    MODEL_PATH = os.environ.get('MODEL_PATH', 'medzome_mvp_model.tflite')
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
        'default': {
            'threshold': 0.5,
            'invert_prediction': False
        }
    }

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
# Image Processing
# ============================================================================

class ImageProcessor:
    """Advanced image processing with lighting and perspective correction"""
    
    @staticmethod
    def preprocess_image(image_data: str, target_height: int = None, target_width: int = None) -> np.ndarray:
        """
        Preprocess image with advanced corrections:
        - Decode base64 image
        - Lighting correction
        - Perspective correction (if needed)
        - Resize to model input size (dynamic dimensions)
        - Normalize
        """
        
        # Use config defaults if not specified
        if target_height is None:
            target_height = config.INPUT_HEIGHT
        if target_width is None:
            target_width = config.INPUT_WIDTH
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply lighting correction
        image = ImageProcessor._correct_lighting(image)
        
        # Detect and correct perspective (optional)
        image = ImageProcessor._detect_and_correct_perspective(image)
        
        # Resize to model input size using dynamic dimensions
        image = cv2.resize(image, (target_width, target_height))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    @staticmethod
    def _correct_lighting(image: np.ndarray) -> np.ndarray:
        """Apply adaptive lighting correction"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge([l, a, b])
            
            # Convert back to RGB
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return image
        except Exception as e:
            print(f"⚠️  Lighting correction failed: {e}")
            return image
    
    @staticmethod
    def _detect_and_correct_perspective(image: np.ndarray) -> np.ndarray:
        """Detect test strip region and correct perspective"""
        try:
            # Simple implementation - can be enhanced with contour detection
            # For now, just return the image
            # In production, implement proper strip detection and perspective transform
            return image
        except Exception as e:
            print(f"⚠️  Perspective correction failed: {e}")
            return image
    
    @staticmethod
    def detect_lines(image: np.ndarray) -> Dict[str, Any]:
        """
        Detect control and test lines in the strip.
        Returns line positions and intensities.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate horizontal intensity profile
            height, width = blurred.shape
            profile = np.mean(blurred, axis=1)
            
            # Find peaks (lines)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(-profile, distance=20, prominence=10)
            
            # Classify peaks as control/test lines
            control_line = None
            test_line = None
            
            if len(peaks) >= 1:
                control_line = {
                    'position': int(peaks[0]),
                    'intensity': float(255 - profile[peaks[0]])
                }
            
            if len(peaks) >= 2:
                test_line = {
                    'position': int(peaks[1]),
                    'intensity': float(255 - profile[peaks[1]])
                }
            
            return {
                'control_line': control_line,
                'test_line': test_line,
                'num_lines': len(peaks)
            }
            
        except Exception as e:
            print(f"⚠️  Line detection failed: {e}")
            return {
                'control_line': None,
                'test_line': None,
                'num_lines': 0
            }

# ============================================================================
# Result Analysis
# ============================================================================

class ResultAnalyzer:
    """Analyze prediction results with semi-quantitative measurement"""
    
    @staticmethod
    def analyze(confidence: float, line_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive result analysis with validation.
        Returns detailed metrics and classifications.
        Marks result as INVALID if control line not detected (likely not a test strip).
        """
        
        # CRITICAL: Check if this is actually a valid test strip
        # A valid lateral flow test MUST have a control line
        has_control_line = line_data.get('control_line') is not None
        
        # If no control line detected, mark as INVALID regardless of confidence
        if not has_control_line:
            return {
                'is_positive': False,
                'is_invalid': True,  # New flag for invalid tests
                'confidence': 0.0,  # Override confidence
                'intensity_score': 0.0,
                'intensity_category': 'Invalid',
                'quality': 'Invalid - Control line not detected. Please ensure image contains a test strip.',
                'line_data': line_data,
                'threshold': config.CONFIDENCE_THRESHOLD,
                'error_message': 'Invalid test strip: Control line not detected. This may not be a test strip image.'
            }
        
        # Valid test strip - proceed with normal analysis
        is_positive = confidence > config.CONFIDENCE_THRESHOLD
        
        # Calculate line intensity score
        intensity_score = ResultAnalyzer._calculate_intensity_score(
            confidence, line_data
        )
        
        # Categorize intensity
        intensity_category = ResultAnalyzer._categorize_intensity(intensity_score)
        
        # Assess quality
        quality = ResultAnalyzer._assess_quality(confidence, line_data)
        
        return {
            'is_positive': is_positive,
            'is_invalid': False,
            'confidence': confidence,
            'intensity_score': intensity_score,
            'intensity_category': intensity_category,
            'quality': quality,
            'line_data': line_data,
            'threshold': config.CONFIDENCE_THRESHOLD
        }
    
    @staticmethod
    def _calculate_intensity_score(confidence: float, line_data: Dict[str, Any]) -> float:
        """Calculate line intensity score (0-100)"""
        
        # Base score from confidence
        score = confidence * 100
        
        # Adjust based on actual test line intensity if available
        if line_data.get('test_line') and line_data['test_line'].get('intensity'):
            test_intensity = line_data['test_line']['intensity']
            # Normalize and blend
            normalized_intensity = (test_intensity / 255.0) * 100
            score = (score * 0.7) + (normalized_intensity * 0.3)
        
        return min(score, 100.0)
    
    @staticmethod
    def _categorize_intensity(score: float) -> str:
        """Categorize intensity into levels"""
        if score >= 80:
            return 'Very Strong'
        elif score >= 60:
            return 'Strong'
        elif score >= 40:
            return 'Moderate'
        elif score >= 20:
            return 'Weak'
        else:
            return 'Very Weak'
    
    @staticmethod
    def _assess_quality(confidence: float, line_data: Dict[str, Any]) -> str:
        """Assess image quality"""
        
        has_control = line_data.get('control_line') is not None
        
        if not has_control:
            return 'Poor - Control line not detected'
        
        if confidence > 0.8:
            return 'Excellent - Clear image'
        elif confidence > 0.6:
            return 'Good - Acceptable quality'
        elif confidence > 0.4:
            return 'Fair - Consider retake'
        else:
            return 'Poor - Retake recommended'

# ============================================================================
# API Endpoints
# ============================================================================

# Global model instance
model_loader = None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    """Serve static files (CSS, JS, images, etc.)"""
    # Exclude API routes - they are handled by specific route handlers
    if path in ['predict', 'health', 'analyze'] or path.startswith('model/'):
        return None
    return send_from_directory('.', path)

def load_model():
    """Load model on server startup"""
    global model_loader
    
    try:
        model_loader = UniversalModelLoader(config.MODEL_PATH)
        print("\n✅ Server ready to serve predictions\n")
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}\n")
        sys.exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader is not None,
        'model_type': model_loader.model_type if model_loader else None,
        'model_path': config.MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Accepts base64 image and returns analysis results.
    """
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = data['image']
        threshold = data.get('threshold', config.CONFIDENCE_THRESHOLD)
        
        # Update threshold if provided
        config.CONFIDENCE_THRESHOLD = threshold
        
        # Start timing
        start_time = time.time()
        
        # Get model's required input dimensions
        height, width, channels = model_loader.get_input_shape()
        
        # Preprocess image with model-specific dimensions
        processed_image = ImageProcessor.preprocess_image(image_data, height, width)
        
        # Run inference
        prediction = model_loader.predict(processed_image)
        
        # Detect lines (for semi-quantitative analysis)
        # Decode image for line detection
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        line_data = ImageProcessor.detect_lines(original_image)
        
        # Analyze results
        analysis = ResultAnalyzer.analyze(prediction['confidence'], line_data)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Build response
        response = {
            'success': True,
            'confidence': analysis['confidence'],
            'is_positive': analysis['is_positive'],
            'is_invalid': analysis.get('is_invalid', False),  # Add invalid flag
            'intensity_score': analysis['intensity_score'],
            'intensity_category': analysis['intensity_category'],
            'quality': analysis['quality'],
            'threshold': analysis['threshold'],
            'control_line_detected': line_data['control_line'] is not None,
            'test_line_detected': line_data['test_line'] is not None,
            'num_lines_detected': line_data['num_lines'],
            'processing_time_ms': processing_time,
            'model_type': model_loader.model_type,
            'timestamp': time.time()
        }
        
        # Add error message if invalid
        if analysis.get('is_invalid'):
            response['error_message'] = analysis.get('error_message', 'Invalid test strip')
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
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
