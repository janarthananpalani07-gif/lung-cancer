import cv2
import numpy as np
from skimage import measure, morphology

# -------------------------------
# 1. Noise Reduction
# -------------------------------
def remove_noise(image):
    # Gaussian Blur
    gaussian = cv2.GaussianBlur(image, (5,5), 0)

    # Median Filter
    median = cv2.medianBlur(gaussian, 5)

    return median


# -------------------------------
# 2. CLAHE Contrast Enhancement
# -------------------------------
def apply_clahe(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)

    return enhanced


# -------------------------------
# 3. Lung Segmentation
# Remove background & keep lungs
# -------------------------------
def segment_lung(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # Morphological closing
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Label connected components
    labels = measure.label(closing)

    # Keep largest connected component (lungs)
    regions = measure.regionprops(labels)

    if len(regions) > 0:
        largest_region = max(regions, key=lambda r: r.area)
        mask = labels == largest_region.label
        mask = mask.astype(np.uint8) * 255
        segmented = cv2.bitwise_and(image, image, mask=mask)
    else:
        segmented = image

    return segmented


# -------------------------------
# 4. Full Preprocessing Pipeline
# -------------------------------
def preprocess_ct_scan(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    noise_removed = remove_noise(image)
    enhanced = apply_clahe(noise_removed)
    segmented = segment_lung(enhanced)

    return segmented
