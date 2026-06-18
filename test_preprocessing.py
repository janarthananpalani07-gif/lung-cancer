import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from preprocessing import remove_noise, apply_clahe, segment_lung, preprocess_ct_scan

# Test image path
test_image = "lung-cancer-app/dataset/LUNA16/images/test/14_jpg.rf.ed2b64b5d3a99303de526281d0393fad.jpg"

print("=" * 60)
print("PREPROCESSING PIPELINE TEST")
print("=" * 60)

# Test 1: Load image
print("\n[1] Loading image...")
if os.path.exists(test_image):
    image = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    print(f"✓ Image loaded successfully!")
    print(f"  Shape: {image.shape}, Dtype: {image.dtype}, Min: {image.min()}, Max: {image.max()}")
else:
    print(f"✗ Image not found at {test_image}")
    exit(1)

# Test 2: Gaussian Filter
print("\n[2] Testing Gaussian Filter...")
try:
    gaussian = cv2.GaussianBlur(image, (5,5), 0)
    print(f"✓ Gaussian filter applied successfully!")
    print(f"  Shape: {gaussian.shape}, Min: {gaussian.min()}, Max: {gaussian.max()}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Median Filter
print("\n[3] Testing Median Filter...")
try:
    median = cv2.medianBlur(gaussian, 5)
    print(f"✓ Median filter applied successfully!")
    print(f"  Shape: {median.shape}, Min: {median.min()}, Max: {median.max()}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: CLAHE Enhancement
print("\n[4] Testing CLAHE Contrast Enhancement...")
try:
    enhanced = apply_clahe(median)
    print(f"✓ CLAHE enhancement applied successfully!")
    print(f"  Shape: {enhanced.shape}, Min: {enhanced.min()}, Max: {enhanced.max()}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Lung Segmentation
print("\n[5] Testing Lung Segmentation...")
try:
    segmented = segment_lung(enhanced)
    print(f"✓ Lung segmentation applied successfully!")
    print(f"  Shape: {segmented.shape}, Min: {segmented.min()}, Max: {segmented.max()}")
    non_zero = np.count_nonzero(segmented)
    print(f"  Non-zero pixels: {non_zero} ({100*non_zero/(segmented.shape[0]*segmented.shape[1]):.2f}%)")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 6: Full Pipeline
print("\n[6] Testing Full Pipeline...")
try:
    result = preprocess_ct_scan(test_image)
    print(f"✓ Full pipeline executed successfully!")
    print(f"  Output shape: {result.shape}")
    print(f"  Output dtype: {result.dtype}")
    print(f"  Output range: [{result.min()}, {result.max()}]")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED!")
print("=" * 60)
