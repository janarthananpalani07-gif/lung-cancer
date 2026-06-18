#!/usr/bin/env python3
import sys
sys.path.insert(0, 'lung-cancer-app')

from pipeline import run_pipeline
import json

# Test with a sample image from the dataset
test_image = "lung-cancer-app/dataset/LUNA16/images/test/63670_JPG_jpg.rf.70cf76f795a0d681d905843f12de0395.jpg"

print(f"Running pipeline on: {test_image}")
print("-" * 50)

try:
    result = run_pipeline(test_image)
    
    # Print results (excluding images)
    print("\nPipeline Results:")
    print(f"  Nodules detected: {result['nodules']}")
    print(f"  Max diameter: {result['max_diameter']:.2f} pixels")
    print(f"  Avg volume: {result['avg_volume']:.2f}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Risk level: {result['risk']}")
    print(f"  Image: {result['image']}")
    print(f"\nDetection Image Path: {result['detected_image_path']}")
    print(f"Detection Image Found: {result['detected_image'] is not None}")
    
    print("\n✓ Pipeline executed successfully!")
    
except Exception as e:
    print(f"✗ Error running pipeline: {e}")
    import traceback
    traceback.print_exc()
