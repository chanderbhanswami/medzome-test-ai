"""
LFA Intensity Normalizer Module
================================
Contains the IntensityNormalizer class for calibrating and normalizing
LFA test strip intensity readings.

This module is used by:
- build_normalization_model.py (to create and save the normalizer)
- final_inference.py (to load and use the normalizer)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from datetime import datetime


def four_parameter_logistic(x, A, B, C, D):
    """
    4-Parameter Logistic (4PL) model - standard for immunoassays.
    A = minimum asymptote (baseline)
    B = Hill's slope (steepness)
    C = inflection point (EC50/IC50)
    D = maximum asymptote
    """
    return D + (A - D) / (1 + (x / C) ** B)


def inverse_4pl(y, A, B, C, D):
    """
    Inverse 4PL - calculate concentration from intensity.
    """
    if y <= A:
        return 0.0
    if y >= D:
        return float('inf')
    
    ratio = (A - D) / (y - D) - 1
    if ratio <= 0:
        return 0.0
    
    return C * (ratio ** (1 / B))


class IntensityNormalizer:
    """
    Normalizes LFA test line intensities based on calibration data.
    
    Key features:
    1. Negative baseline correction - sets negative samples to 0
    2. Positive sample normalization using fitted curve
    3. Concentration estimation from intensity
    """
    
    def __init__(self):
        self.is_fitted = False
        self.negative_threshold = 0.0
        self.negative_mean = 0.0
        self.negative_std = 0.0
        self.positive_baseline = 0.0
        self.tc_ratio_threshold = 0.0
        self.normalization_scale = 1.0
        
        # Additional stats
        self.neg_control_mean = 0.0
        self.neg_tc_ratio_mean = 0.0
        self.neg_tc_ratio_std = 0.0
        
        # Classification model
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
        # Calibration curve parameters (4PL)
        self.calibration_params = None
        self.calibration_r2 = 0.0
        
        # Statistics
        self.stats = {}
    
    def fit(self, df):
        """
        Fit the normalizer using extracted intensity data.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Must contain: Ground_Truth, Test_Intensity, Control_Intensity, 
                         Concentration_ng, TC_Ratio
        """
        print("\n" + "="*70)
        print("📊 FITTING INTENSITY NORMALIZATION MODEL")
        print("="*70)
        
        # Separate positive and negative samples
        negative_samples = df[df['Ground_Truth'] == 'Negative'].copy()
        positive_samples = df[df['Ground_Truth'] == 'Positive'].copy()
        
        print(f"\n📈 Data Summary:")
        print(f"   Total samples: {len(df)}")
        print(f"   Negative samples (0 ng/mL): {len(negative_samples)}")
        print(f"   Positive samples (>0 ng/mL): {len(positive_samples)}")
        
        # =====================================================================
        # STEP 1: Analyze Negative Baseline
        # =====================================================================
        print("\n🔬 Step 1: Analyzing Negative Baseline...")
        
        neg_intensities = negative_samples['Test_Intensity'].values
        self.negative_mean = np.mean(neg_intensities)
        self.negative_std = np.std(neg_intensities)
        
        # Set threshold at mean + 2*std (95% confidence)
        self.negative_threshold = self.negative_mean + 2 * self.negative_std
        
        # Also calculate control line stats for ratio-based detection
        neg_control = negative_samples['Control_Intensity'].values
        neg_tc_ratio = negative_samples['TC_Ratio'].values
        
        self.neg_control_mean = np.mean(neg_control)
        self.neg_tc_ratio_mean = np.mean(neg_tc_ratio)
        self.neg_tc_ratio_std = np.std(neg_tc_ratio)
        self.tc_ratio_threshold = self.neg_tc_ratio_mean + 2 * self.neg_tc_ratio_std
        
        print(f"   Negative Test Intensity: {self.negative_mean:.3f} ± {self.negative_std:.3f}")
        print(f"   Negative TC Ratio: {self.neg_tc_ratio_mean:.4f} ± {self.neg_tc_ratio_std:.4f}")
        print(f"   Detection Threshold (intensity): {self.negative_threshold:.3f}")
        print(f"   Detection Threshold (TC ratio): {self.tc_ratio_threshold:.4f}")
        
        # =====================================================================
        # STEP 2: Train Classification Model
        # =====================================================================
        print("\n🤖 Step 2: Training Classification Model...")
        
        # Prepare features for classification
        X = df[['Test_Intensity', 'Control_Intensity', 'TC_Ratio']].values
        y = (df['Ground_Truth'] == 'Positive').astype(int).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train logistic regression
        self.classifier = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.classifier.fit(X_scaled, y)
        
        # Calculate accuracy
        predictions = self.classifier.predict(X_scaled)
        accuracy = np.mean(predictions == y)
        print(f"   Classification Accuracy: {accuracy*100:.1f}%")
        
        # Feature importance
        coefs = self.classifier.coef_[0]
        features = ['Test_Intensity', 'Control_Intensity', 'TC_Ratio']
        print(f"   Feature Importance:")
        for feat, coef in zip(features, coefs):
            print(f"      {feat}: {coef:.4f}")
        
        # =====================================================================
        # STEP 3: Fit Calibration Curve (Concentration vs Intensity)
        # =====================================================================
        print("\n📈 Step 3: Fitting Calibration Curve...")
        
        if len(positive_samples) >= 4:
            # Group by concentration and get mean intensity
            conc_groups = positive_samples.groupby('Concentration_ng').agg({
                'Test_Intensity': ['mean', 'std', 'count'],
                'TC_Ratio': ['mean', 'std']
            }).reset_index()
            
            conc_groups.columns = ['Concentration', 'Intensity_Mean', 'Intensity_Std', 
                                   'Count', 'Ratio_Mean', 'Ratio_Std']
            
            print(f"\n   Concentration-Intensity Relationship:")
            print(f"   {'Conc (ng/mL)':<15} {'Intensity':<15} {'TC Ratio':<15} {'n':<5}")
            print(f"   {'-'*50}")
            
            # Add negative baseline
            print(f"   {'0.0 (neg)':<15} {self.negative_mean:<15.3f} {self.neg_tc_ratio_mean:<15.4f} {len(negative_samples):<5}")
            
            for _, row in conc_groups.iterrows():
                print(f"   {row['Concentration']:<15.2f} {row['Intensity_Mean']:<15.3f} "
                      f"{row['Ratio_Mean']:<15.4f} {int(row['Count']):<5}")
            
            # Try to fit 4PL curve
            try:
                x_data = np.array([0] + conc_groups['Concentration'].tolist())
                y_data = np.array([self.negative_mean] + conc_groups['Intensity_Mean'].tolist())
                
                # Initial guess for 4PL: A=baseline, B=1, C=median_conc, D=max
                p0 = [self.negative_mean, 1.0, np.median(x_data[1:]), np.max(y_data)]
                
                # Fit with bounds
                bounds = (
                    [0, 0.1, 0.01, y_data.max()*0.5],  # lower bounds
                    [y_data.max(), 10, 100, y_data.max()*2]  # upper bounds
                )
                
                popt, pcov = curve_fit(four_parameter_logistic, x_data, y_data, 
                                      p0=p0, bounds=bounds, maxfev=5000)
                
                self.calibration_params = {
                    'A': popt[0],  # baseline
                    'B': popt[1],  # slope
                    'C': popt[2],  # EC50
                    'D': popt[3]   # maximum
                }
                
                # Calculate R²
                y_pred = four_parameter_logistic(x_data, *popt)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                self.calibration_r2 = 1 - (ss_res / ss_tot)
                
                print(f"\n   4PL Curve Fitted Successfully!")
                print(f"   Parameters: A={popt[0]:.3f}, B={popt[1]:.3f}, C={popt[2]:.3f}, D={popt[3]:.3f}")
                print(f"   R² = {self.calibration_r2:.4f}")
                
            except Exception as e:
                print(f"\n   ⚠️  Could not fit 4PL curve: {e}")
                print(f"   Using linear interpolation instead.")
                self.calibration_params = None
        else:
            print(f"   ⚠️  Not enough positive concentration levels for calibration curve")
            self.calibration_params = None
        
        # =====================================================================
        # STEP 4: Calculate Normalization Parameters
        # =====================================================================
        print("\n📐 Step 4: Calculating Normalization Parameters...")
        
        # For positive samples, subtract baseline
        pos_intensities = positive_samples['Test_Intensity'].values
        self.positive_baseline = self.negative_mean
        
        # Normalized positive range
        normalized_positive = pos_intensities - self.positive_baseline
        normalized_positive = np.maximum(normalized_positive, 0)  # Clip to 0
        
        self.normalization_scale = np.max(normalized_positive) if len(normalized_positive) > 0 else 1.0
        
        print(f"   Baseline to subtract: {self.positive_baseline:.3f}")
        print(f"   Normalization scale: {self.normalization_scale:.3f}")
        
        # Store comprehensive stats
        self.stats = {
            'negative': {
                'mean_intensity': float(self.negative_mean),
                'std_intensity': float(self.negative_std),
                'threshold': float(self.negative_threshold),
                'mean_tc_ratio': float(self.neg_tc_ratio_mean),
                'std_tc_ratio': float(self.neg_tc_ratio_std),
                'count': len(negative_samples)
            },
            'positive': {
                'min_intensity': float(pos_intensities.min()) if len(pos_intensities) > 0 else 0,
                'max_intensity': float(pos_intensities.max()) if len(pos_intensities) > 0 else 0,
                'mean_intensity': float(pos_intensities.mean()) if len(pos_intensities) > 0 else 0,
                'count': len(positive_samples)
            },
            'calibration': {
                'fitted': self.calibration_params is not None,
                'r2': float(self.calibration_r2) if self.calibration_params else 0,
                'params': self.calibration_params
            },
            'classification_accuracy': float(accuracy),
            'fitted_date': datetime.now().isoformat()
        }
        
        self.is_fitted = True
        print("\n✅ Normalization model fitted successfully!")
        
        return self
    
    def normalize(self, test_intensity, control_intensity, ai_prediction=None, ai_confidence=None):
        """
        Normalize a single intensity reading.
        
        Parameters:
        -----------
        test_intensity : float
            Raw test line intensity
        control_intensity : float
            Raw control line intensity  
        ai_prediction : str, optional
            AI model prediction ('Positive' or 'Negative')
        ai_confidence : float, optional
            AI model confidence (0-1)
            
        Returns:
        --------
        dict with:
            - normalized_intensity: Normalized test intensity (0 for negative)
            - is_positive: Boolean classification
            - confidence: Confidence score
            - estimated_concentration: Estimated concentration (if calibrated)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Calculate TC ratio
        tc_ratio = test_intensity / control_intensity if control_intensity > 0 else 0
        
        # Intensity-based decision
        intensity_positive = test_intensity > self.negative_threshold
        ratio_positive = tc_ratio > self.tc_ratio_threshold
        
        # Classifier-based decision
        X = np.array([[test_intensity, control_intensity, tc_ratio]])
        X_scaled = self.scaler.transform(X)
        classifier_pred = self.classifier.predict(X_scaled)[0]
        classifier_prob = self.classifier.predict_proba(X_scaled)[0][1]
        
        # Combine decisions - priority: AI prediction > Classifier > Threshold
        if ai_prediction is not None:
            ai_says_positive = ai_prediction.lower() == 'positive'
            
            if ai_says_positive and ai_confidence and ai_confidence > 0.7:
                is_positive = True
            elif not ai_says_positive and ai_confidence and ai_confidence > 0.7:
                is_positive = False
            else:
                is_positive = classifier_pred == 1
        else:
            is_positive = classifier_pred == 1 or (intensity_positive and ratio_positive)
        
        # Calculate Normalized Intensity
        if not is_positive:
            # NEGATIVE: Force intensity to 0
            normalized_intensity = 0.0
            estimated_concentration = 0.0
            confidence = 1.0 - classifier_prob
        else:
            # POSITIVE: Normalize intensity
            normalized_intensity = test_intensity - self.positive_baseline
            normalized_intensity = max(0, normalized_intensity)
            
            # Scale to 0-100 range
            if self.normalization_scale > 0:
                normalized_intensity = (normalized_intensity / self.normalization_scale) * 100
            
            # Estimate concentration if calibration available
            if self.calibration_params is not None:
                try:
                    estimated_concentration = inverse_4pl(
                        test_intensity,
                        self.calibration_params['A'],
                        self.calibration_params['B'],
                        self.calibration_params['C'],
                        self.calibration_params['D']
                    )
                    estimated_concentration = min(estimated_concentration, 100.0)
                except:
                    estimated_concentration = None
            else:
                estimated_concentration = None
            
            confidence = classifier_prob
        
        return {
            'normalized_intensity': round(normalized_intensity, 2),
            'is_positive': is_positive,
            'confidence': round(confidence, 4),
            'estimated_concentration': round(estimated_concentration, 3) if estimated_concentration else None,
            'raw_intensity': round(test_intensity, 2),
            'tc_ratio': round(tc_ratio, 4)
        }

    def save(self, model_path='./models/intensity_normalizer.pkl', 
             json_path='./models/normalization_params.json'):
        """Save the fitted model."""
        import os
        import pickle
        import json
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save pickle model
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Save JSON params (for web interface / other languages)
        json_data = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'thresholds': {
                'negative_intensity_threshold': self.negative_threshold,
                'tc_ratio_threshold': self.tc_ratio_threshold,
                'baseline': self.positive_baseline
            },
            'negative_stats': {
                'mean': self.negative_mean,
                'std': self.negative_std
            },
            'calibration': self.calibration_params,
            'calibration_r2': self.calibration_r2,
            'normalization_scale': self.normalization_scale,
            'stats': self.stats
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"💾 Parameters saved to: {json_path}")
    
    @classmethod
    def load(cls, model_path='./models/intensity_normalizer.pkl'):
        """Load a fitted model."""
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)
