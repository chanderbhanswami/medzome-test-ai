import cv2
import numpy as np
from scipy.signal import find_peaks

class LFAQuantifier:
    def __init__(self):
        pass

    def process_strip(self, image_path):
        img = cv2.imread(image_path)
        if img is None: 
            return {"error": "Image not found", "result": "Error"}

        h, w = img.shape[:2]
        
        # 1. Wider Crop (30% to 70%) to ensure we don't miss off-center lines
        center_crop = img[10:h-10, int(w*0.30):int(w*0.70)]
        
        # 2. Invert Green Channel (Lines become bright)
        b, g, r = cv2.split(center_crop)
        signal = 255 - g 
        
        # 3. Create Profile
        profile = np.mean(signal, axis=1)

        # 4. Find Control Line (The strongest peak in top half)
        # We assume Control is in the top 60% of the image
        top_half_profile = profile[:int(h*0.6)]
        
        results = {
            "control_intensity": 0.0,
            "test_intensity": 0.0,
            "ratio": 0.0,
            "result": "Invalid"
        }

        # Find Control Peak
        c_peaks, _ = find_peaks(top_half_profile, height=20, distance=20)
        
        if len(c_peaks) > 0:
            # Pick the tallest peak as Control
            c_pos = c_peaks[np.argmax(top_half_profile[c_peaks])]
            c_intensity = float(profile[c_pos])
            results['control_intensity'] = c_intensity
            
            # 5. Targeted Test Line Search
            # The Test line is physically located below the Control line.
            # In a 384px image, it's typically ~80-140 pixels below.
            
            start_search = c_pos + 60
            end_search = min(c_pos + 160, h-10)
            
            if start_search < end_search:
                test_zone = profile[start_search:end_search]
                
                # Instead of find_peaks (which misses faint blurs), 
                # we just look for the MAXIMUM signal in this zone.
                max_signal_idx = np.argmax(test_zone)
                max_signal = test_zone[max_signal_idx]
                
                # To be a valid line, it must be notably higher than the "background noise"
                # Calculate local background (min value in the zone)
                background = np.min(test_zone)
                true_intensity = float(max_signal - background) # Peak relative to noise
                
                results['test_intensity'] = max(0.0, true_intensity)

                # Decision Logic
                if results['test_intensity'] > 3.0: # Threshold for "Visible"
                    results['result'] = "Positive"
                else:
                    results['result'] = "Negative"
                    
                if c_intensity > 0:
                    results['ratio'] = results['test_intensity'] / c_intensity

        return results