import json
import numpy as np
from analyze_punch import analyze_punch_video, calculate_angle
import os

def create_reference_database():
    """Analyze professional technique videos to create benchmarks"""
    
    reference_metrics = {
        "jab_extension_time": {
            "min_frames": 8,  # Minimum frames for extension
            "max_frames": 15, # Maximum frames for extension  
            "optimal_range": [10, 12]
        },
        "hip_rotation": {
            "min_angle": 15,  # Minimum hip rotation degrees
            "optimal_range": [25, 35]
        },
        "arm_extension": {
            "full_extension_threshold": 160,  # Degrees for full extension
            "optimal_range": [165, 180]
        },
        "return_speed": {
            "max_return_frames": 12,  # Frames to return to guard
            "optimal_range": [8, 10]
        }
    }
    
    return reference_metrics

def save_reference_data():
    """Save reference metrics to file"""
    ref_data = create_reference_database()
    with open('reference_metrics.json', 'w') as f:
        json.dump(ref_data, f, indent=2)
    print("Reference metrics saved!")

if __name__ == "__main__":
    save_reference_data()