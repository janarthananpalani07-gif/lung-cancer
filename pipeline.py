from matplotlib.pyplot import gray
from ultralytics import YOLO 
import cv2
from PIL import Image
from utils import risk_level
import os
import numpy as np

# ================= RESNET LOADING =================
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

# --------------------------------------------------
# 📂 PATH SETUP
# --------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
yolo_weights_path = os.path.join(script_dir, "best.pt")
CANCER_CLASS_INDEX = 1
NON_CANCER_CLASS_INDEX = 0

# ----------------------------------------------------------
# 🎯 LOAD YOLO TUMOR DETECTOR
# ----------------------------------------------------------
yolo = YOLO(yolo_weights_path)

# --------------------------------------------------
# 🔥 LOAD NEW CT RESNET CLASSIFIER
# --------------------------------------------------

device = torch.device("cpu")

classifier = models.resnet50(weights=None)
classifier.fc = nn.Sequential(
    nn.Linear(classifier.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)

# ⭐ IMPORTANT: load NEW CT model (graceful fallback if not yet trained)
resnet_path = os.path.join(script_dir, "models/resnet_ct_cancer.pth")
CLASSIFIER_READY = False

if os.path.exists(resnet_path):
    classifier.load_state_dict(torch.load(resnet_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    CLASSIFIER_READY = True
    print("✅ ResNet classifier loaded.")
else:
    print("⚠️  WARNING: models/resnet_ct_cancer.pth not found. "
          "Running in YOLO-only mode — run train_resnet.py to enable classification.")

# --------------------------------------------------
# 🎯 RESNET TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])

# --------------------------------------------------
# 🧠 PREPROCESS CT
# --------------------------------------------------

def preprocess_ct_for_models(image_path):
    img = cv2.imread(image_path)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = cv2.GaussianBlur(gray, (5,5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb

# --------------------------------------------------
# 🫁 LUNG SEGMENTATION
# --------------------------------------------------

def segment_lungs(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return segmented, mask


# --------------------------------------------------
# 🎯 TUMOR SEGMENTATION
# --------------------------------------------------

def segment_tumor_irregular(crop_rgb):
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (9, 9), 0)
    high_density = cv2.subtract(enhanced, blur)

    threshold_value = max(10, int(np.percentile(high_density, 90)))
    _, thresh = cv2.threshold(high_density, threshold_value, 255, cv2.THRESH_BINARY)

    border = max(4, min(crop_rgb.shape[:2]) // 15)
    thresh[:border, :] = 0
    thresh[-border:, :] = 0
    thresh[:, :border] = 0
    thresh[:, -border:] = 0

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray, dtype=np.uint8)
    crop_h, crop_w = gray.shape
    crop_area = crop_h * crop_w
    crop_center = np.array([crop_w / 2.0, crop_h / 2.0], dtype=np.float32)

    best_contour = None
    best_score = -1.0
    fallback_contour = None
    fallback_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(40, crop_area * 0.002) or area > crop_area * 0.45:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        touches_edge = x <= 1 or y <= 1 or (x + w) >= crop_w - 1 or (y + h) >= crop_h - 1
        if touches_edge:
            continue

        if area > fallback_area:
            fallback_contour = contour
            fallback_area = area

        extent = area / (w * h + 1e-6)
        if extent > 0.9:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        centroid = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )
        center_distance = np.linalg.norm(
            (centroid - crop_center) / np.array([crop_w, crop_h], dtype=np.float32)
        )
        center_score = 1.0 - min(1.0, center_distance * 1.8)

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        shape_score = 1.0 - min(1.0, abs(circularity - 0.45))

        score = area * (0.6 + 0.4 * center_score) * max(0.2, shape_score)
        if score > best_score:
            best_contour = contour
            best_score = score

    chosen_contour = best_contour if best_contour is not None else fallback_contour
    if chosen_contour is not None:
        cv2.drawContours(mask, [chosen_contour], -1, 255, -1)

    tumor_segmented = cv2.bitwise_and(crop_rgb, crop_rgb, mask=mask)
    return tumor_segmented, mask

def crop_nodule_region(original_image, yolo_boxes):

    h, w, _ = original_image.shape

    if len(yolo_boxes) == 0:
        print("⚠️ No YOLO boxes → center crop fallback")

        cx, cy = w // 2, h // 2
        size = min(h, w) // 4

        x1 = cx - size
        y1 = cy - size
        x2 = cx + size
        y2 = cy + size

    else:
        # ⭐ pick BEST confidence box
        best_box = max(yolo_boxes, key=lambda b: b[4])
        x1, y1, x2, y2, conf = best_box

        print("Using YOLO box:", best_box)

        pad = 40
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad

    # clamp again
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    crop = original_image[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        print("🚨 FINAL fallback to full image")
        crop = original_image.copy()

    return crop


def get_best_nodule_crop_and_box(image_rgb, yolo_boxes, pad=40):
    h, w = image_rgb.shape[:2]

    if not yolo_boxes:
        cx, cy = w // 2, h // 2
        size = min(h, w) // 4
        x1 = cx - size
        y1 = cy - size
        x2 = cx + size
        y2 = cy + size
    else:
        best_box = max(yolo_boxes, key=lambda b: b[4])
        x1, y1, x2, y2, _ = best_box
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    crop = image_rgb[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return image_rgb.copy(), (0, 0, w, h)

    return crop, (x1, y1, x2, y2)

# --------------------------------------------------
# 🎯 DRAW BOUNDING BOX
# --------------------------------------------------

def draw_bbox(image, box, mask_crop=None):

    img = image.copy()
    x1,y1,x2,y2 = map(int, box)

    # If we have tumor mask → draw REAL tumor contour
    if mask_crop is not None:

        contours,_ = cv2.findContours(mask_crop,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:

            # shift contour back to original image coords
            cnt = cnt + np.array([[x1, y1]])

            # 🔴 draw irregular tumor boundary
            cv2.drawContours(img, [cnt], -1, (0,0,255), 2)

    return img

# --------------------------------------------------
# 🔥 HEATMAP FROM TUMOR REGION (Pseudo GradCAM)
# --------------------------------------------------

def create_heatmap(crop_rgb):
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    heatmap = cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def build_overlay_activation_mask(cam, lung_mask=None, tumor_mask=None, focus_mask=None):
    height, width = cam.shape[:2]

    if lung_mask is None:
        lung_mask_resized = np.full((height, width), 255, dtype=np.uint8)
    else:
        lung_mask_resized = cv2.resize(lung_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        lung_mask_resized = np.where(lung_mask_resized > 0, 255, 0).astype(np.uint8)

    for candidate_mask in (focus_mask, tumor_mask):
        if candidate_mask is None:
            continue

        mask_resized = cv2.resize(candidate_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_resized = np.where(mask_resized > 0, 255, 0).astype(np.uint8)
        mask_resized = cv2.bitwise_and(mask_resized, lung_mask_resized)
        if np.count_nonzero(mask_resized) > 0:
            return mask_resized

    lung_values = cam[lung_mask_resized > 0]
    threshold = 0.45
    if lung_values.size > 0:
        threshold = float(np.percentile(lung_values, 92))
        threshold = min(0.75, max(0.35, threshold))

    activation_mask = np.zeros((height, width), dtype=np.uint8)
    activation_mask[(cam >= threshold) & (lung_mask_resized > 0)] = 255

    if np.count_nonzero(activation_mask) < 40:
        relaxed_threshold = max(0.25, threshold - 0.10)
        activation_mask[(cam >= relaxed_threshold) & (lung_mask_resized > 0)] = 255

    kernel = np.ones((5, 5), np.uint8)
    activation_mask = cv2.morphologyEx(activation_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    activation_mask = cv2.morphologyEx(activation_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if np.count_nonzero(activation_mask) == 0:
        activation_mask[(cam >= 0.20) & (lung_mask_resized > 0)] = 255

    return activation_mask


def build_reference_overlay_base(image_rgb, size=(224, 224)):
    ct_view = cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(ct_view, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.normalize(gray, None, 18, 248, cv2.NORM_MINMAX)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def build_tumor_overlay_view(
    image_rgb,
    focus_mask,
    size=(224, 224),
    zoom_scale=6.0,
    show_crosshair=False,
    min_side_ratio=0.72,
):
    if focus_mask is None or np.count_nonzero(focus_mask) == 0:
        return cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)

    mask_uint8 = np.where(focus_mask > 0, 255, 0).astype(np.uint8)
    points = cv2.findNonZero(mask_uint8)
    x, y, w, h = cv2.boundingRect(points)

    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    side = int(max(w, h) * zoom_scale)
    min_side = int(min(image_rgb.shape[:2]) * min_side_ratio)
    side = max(side, min_side, 72)
    side = min(side, max(image_rgb.shape[:2]))

    half_side = side / 2.0
    x1 = int(round(center_x - half_side))
    y1 = int(round(center_y - half_side))
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image_rgb.shape[1]:
        shift = x2 - image_rgb.shape[1]
        x1 = max(0, x1 - shift)
        x2 = image_rgb.shape[1]
    if y2 > image_rgb.shape[0]:
        shift = y2 - image_rgb.shape[0]
        y1 = max(0, y1 - shift)
        y2 = image_rgb.shape[0]

    crop = image_rgb[y1:y2, x1:x2]
    view = resize_with_padding(crop, size, background=(0, 0, 0))
    if show_crosshair:
        center = (size[0] // 2, size[1] // 2)
        cv2.line(view, (center[0], 0), (center[0], size[1] - 1), (126, 255, 180), 1, cv2.LINE_AA)
        cv2.line(view, (0, center[1]), (size[0] - 1, center[1]), (255, 240, 150), 1, cv2.LINE_AA)
    return view


def build_single_color_tumor_overlay(base_image_rgb, focus_mask, size=(224, 224), fill_color=(255, 72, 72)):
    if focus_mask is None or np.count_nonzero(focus_mask) == 0:
        return cv2.resize(base_image_rgb, size, interpolation=cv2.INTER_AREA)

    mask_uint8 = np.where(focus_mask > 0, 255, 0).astype(np.uint8)
    color_overlay = base_image_rgb.copy().astype(np.float32)
    color_layer = np.zeros_like(base_image_rgb, dtype=np.float32)
    color_layer[:, :] = np.array(fill_color, dtype=np.float32)

    alpha = np.zeros(mask_uint8.shape, dtype=np.float32)
    alpha[mask_uint8 > 0] = 0.70
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    color_overlay = color_overlay * (1.0 - alpha[..., None]) + color_layer * alpha[..., None]
    color_overlay = np.clip(color_overlay, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(color_overlay, contours, -1, fill_color, 1)

    return build_tumor_overlay_view(
        color_overlay,
        mask_uint8,
        size=size,
        zoom_scale=4.0,
        show_crosshair=False,
        min_side_ratio=0.52,
    )


def build_zoomed_heatmap_view(relevance_map, shape_mask, size):
    mask_uint8 = np.where(shape_mask > 0, 255, 0).astype(np.uint8)
    heatmap_height, heatmap_width = relevance_map.shape[:2]
    masked_heatmap = np.zeros((heatmap_height, heatmap_width, 3), dtype=np.uint8)

    if np.count_nonzero(mask_uint8) == 0:
        return masked_heatmap

    masked_relevance = relevance_map * (mask_uint8 > 0).astype(np.float32)
    if np.count_nonzero(masked_relevance) > 0:
        masked_values = masked_relevance[mask_uint8 > 0]
        min_value = float(masked_values.min())
        max_value = float(masked_values.max())
        masked_relevance = (masked_relevance - min_value) / max(max_value - min_value, 1e-6)

    shape_core = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    if shape_core.max() > 0:
        shape_core = shape_core / shape_core.max()

    shaped_relevance = (0.55 * shape_core) + (0.45 * masked_relevance)
    if shaped_relevance.max() > 0:
        shaped_relevance = shaped_relevance / shaped_relevance.max()

    color_heatmap = cv2.applyColorMap(np.uint8(shaped_relevance * 255), cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    masked_heatmap[mask_uint8 > 0] = color_heatmap[mask_uint8 > 0]

    points = cv2.findNonZero(mask_uint8)
    x, y, w, h = cv2.boundingRect(points)
    pad_x = max(8, int(w * 0.35))
    pad_y = max(8, int(h * 0.35))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(masked_heatmap.shape[1], x + w + pad_x)
    y2 = min(masked_heatmap.shape[0], y + h + pad_y)

    zoom_crop = masked_heatmap[y1:y2, x1:x2]
    return resize_with_padding(zoom_crop, size)


def build_heatmap_overlay(base_image_rgb, cam, lung_mask=None, tumor_mask=None, focus_mask=None):
    height, width = base_image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    activation_mask = build_overlay_activation_mask(
        cam_resized,
        lung_mask=lung_mask,
        tumor_mask=tumor_mask,
        focus_mask=focus_mask,
    )

    seed_strength = activation_mask.astype(np.float32) / 255.0
    soft_seed = cv2.GaussianBlur(seed_strength, (0, 0), sigmaX=16, sigmaY=16)
    if soft_seed.max() > 0:
        soft_seed = soft_seed / soft_seed.max()

    core_seed = np.zeros_like(seed_strength, dtype=np.float32)
    if np.count_nonzero(activation_mask) > 0:
        core_seed = cv2.distanceTransform(activation_mask, cv2.DIST_L2, 5)
        if core_seed.max() > 0:
            core_seed = core_seed / core_seed.max()
        core_seed = cv2.GaussianBlur(core_seed, (0, 0), sigmaX=6, sigmaY=6)

    gated_cam = cam_resized * (0.20 + (0.80 * soft_seed))
    relevance_map = (0.18 * gated_cam) + (0.50 * soft_seed) + (0.32 * core_seed)

    if lung_mask is not None:
        lung_mask_resized = cv2.resize(lung_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        lung_mask_resized = lung_mask_resized > 0
        relevance_map = relevance_map * lung_mask_resized.astype(np.float32)
    else:
        lung_mask_resized = np.ones((height, width), dtype=bool)

    if relevance_map.max() > 0:
        relevance_map = relevance_map / relevance_map.max()

    color_heatmap = cv2.applyColorMap(np.uint8(relevance_map * 255), cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    heatmap_shape_mask = activation_mask > 0
    display_mask = ((relevance_map > 0.08) | heatmap_shape_mask) & lung_mask_resized
    masked_heatmap = build_zoomed_heatmap_view(
        relevance_map,
        heatmap_shape_mask,
        size=(width, height),
    )

    alpha = np.where(
        display_mask,
        0.18 + (0.62 * np.power(relevance_map, 0.85)),
        0.0,
    ).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (9, 9), 0)

    overlay = base_image_rgb.astype(np.float32)
    overlay = overlay * (1.0 - alpha[..., None]) + color_heatmap.astype(np.float32) * alpha[..., None]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    view_focus_mask = None
    for candidate_mask in (focus_mask, tumor_mask, activation_mask):
        if candidate_mask is not None and np.count_nonzero(candidate_mask) > 0:
            view_focus_mask = candidate_mask
            break

    overlay = build_tumor_overlay_view(overlay, view_focus_mask, size=(width, height))

    return masked_heatmap, overlay


# --------------------------------------------------
# 🧠 HYBRID DECISION ENGINE
# --------------------------------------------------

def hybrid_cancer_decision(cancer_prob, diameter, yolo_conf):
    score = 0
    score += cancer_prob * 40

    if diameter > 120: score += 25
    elif diameter > 80: score += 18
    elif diameter > 40: score += 10
    else: score += 3

    score += yolo_conf * 20
    score += 8  # medical prior

    return score / 100

# --------------------------------------------------
# 🔥 GRADCAM FOR TUMOR LOCALIZATION
# --------------------------------------------------

features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# hook last conv layer of ResNet

target_layer = classifier.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# --------------------------------------------------
# 🔥 GRADCAM FOR TUMOR LOCALIZATION (FIXED VERSION)
# --------------------------------------------------

def generate_gradcam(input_tensor):
    # --- fallback: model not trained ---
    if not CLASSIFIER_READY:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        return blank, np.zeros((224, 224), dtype=np.float32)

    classifier.eval()

    gradients = []
    activations = []

    # Hook the LAST CONV LAYER of ResNet
    target_layer = classifier.layer4[2].conv3

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Register hooks
    fwd_hook = target_layer.register_forward_hook(forward_hook)
    bwd_hook = target_layer.register_full_backward_hook(backward_hook)

    # --------------------------------------------------
    # Forward pass (NO GRAD TRACKING EXCEPT TARGET)
    # --------------------------------------------------
    input_tensor.requires_grad = True
    output = classifier(input_tensor)

    cancer_score = output[0, CANCER_CLASS_INDEX]

    classifier.zero_grad()
    cancer_score.backward()

    # Remove hooks
    fwd_hook.remove()
    bwd_hook.remove()

    # --------------------------------------------------
    # Convert tensors safely → numpy
    # --------------------------------------------------
    grads = gradients[0].cpu().numpy()[0]      # [C,H,W]
    fmap  = activations[0].cpu().numpy()[0]    # [C,H,W]

    # --------------------------------------------------
    # GradCAM weights
    # --------------------------------------------------
    weights = np.mean(grads, axis=(1,2))

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)

    # --------------------------------------------------
    # HIGH QUALITY UPSCALING (fix bad heatmap!!)
    # --------------------------------------------------
    cam = cv2.resize(cam, (224,224), interpolation=cv2.INTER_CUBIC)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Convert to heatmap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap, cam

# --------------------------------------------------
# 🫁 Lung mask at 224
# --------------------------------------------------
def get_lung_mask_224(lung_img):
    gray = cv2.cvtColor(lung_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask,5)
    return mask

# --------------------------------------------------
# 🎯 CT suspicious intensity map
# --------------------------------------------------
def ct_intensity_map(lung_img):
    gray = cv2.cvtColor(lung_img, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm,(7,7),0)
    high_density = cv2.subtract(norm, blur)
    return high_density

# --------------------------------------------------
# 🧬 Texture edges (tumor irregularity)
# --------------------------------------------------
def texture_edges(lung_img):
    gray = cv2.cvtColor(lung_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,40,120)
    edges = cv2.dilate(edges,np.ones((3,3),np.uint8))
    return edges

# --------------------------------------------------
# 🎯 YOLO NODULE DETECTION (FINAL FIX)
# --------------------------------------------------
from ultralytics import YOLO
import numpy as np

yolo_model = YOLO(yolo_weights_path)

def detect_tumor_yolo(image):

    h0, w0 = image.shape[:2]

    results = yolo_model.predict(
        source=image,
        conf=0.05,
        imgsz=640,
        verbose=False
    )

    boxes_out = []

    r = results[0]

    if r.boxes is None:
        print("YOLO detected boxes: 0")
        return []

    # YOLO resized shape
    h1, w1 = r.orig_shape   # ⭐ THIS IS THE FIX

    scale_x = w0 / w1
    scale_y = h0 / h1

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), conf in zip(boxes, confs):

        # convert back to original image coordinates
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # clamp to image bounds (VERY IMPORTANT)
        x1 = max(0, min(x1, w0-1))
        y1 = max(0, min(y1, h0-1))
        x2 = max(0, min(x2, w0-1))
        y2 = max(0, min(y2, h0-1))

        # ignore ultra tiny boxes
        if (x2 - x1) < 15 or (y2 - y1) < 15:
            continue

        boxes_out.append([x1, y1, x2, y2, float(conf)])

    print("YOLO detected boxes:", len(boxes_out))
    return boxes_out
# --------------------------------------------------
# 🫁 Apply Lung Mask
# --------------------------------------------------
def apply_lung_mask(image):
    """
    Simple lung region extraction using thresholding
    Returns:
        lung_only_image,
        binary_mask
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Basic threshold (can improve later)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    mask = cv2.medianBlur(mask, 5)

    # Convert mask to 3-channel
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    lung_only = cv2.bitwise_and(image, mask_3ch)

    return lung_only, mask

def preprocess_for_yolo(ct_img):
    # 1️⃣ convert to grayscale
    gray = cv2.cvtColor(ct_img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ histogram equalization (VERY IMPORTANT)
    gray = cv2.equalizeHist(gray)

    # 3️⃣ CLAHE contrast boost (medical CT style)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 4️⃣ convert back to 3-channel
    img3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 5️⃣ resize EXACTLY like training
    img3 = cv2.resize(img3, (640, 640))

    return img3


def preprocess_for_yolo_same_size(ct_img):
    gray = cv2.cvtColor(ct_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def classify_ct_slice(image_rgb):
    # --- fallback: model not trained yet ---
    if not CLASSIFIER_READY:
        blank_224 = cv2.resize(image_rgb, (224, 224))
        dummy_tensor = torch.zeros(1, 3, 224, 224)
        return "Non-Cancer", 0.0, dummy_tensor, blank_224

    image_224 = cv2.resize(image_rgb, (224, 224))
    image_pil = Image.fromarray(image_224)
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    classifier.eval()
    with torch.no_grad():
        outputs = classifier(input_tensor)

    probs = torch.softmax(outputs, dim=1)
    cancer_prob = float(probs[0][CANCER_CLASS_INDEX])
    non_cancer_prob = float(probs[0][NON_CANCER_CLASS_INDEX])
    prediction = "Cancer" if cancer_prob >= non_cancer_prob else "Non-Cancer"

    return prediction, cancer_prob, input_tensor, image_224


def build_detection_preview(image_rgb, yolo_boxes, focus_mask=None, size=(224, 224)):
    preview_source = image_rgb.copy()
    preview = cv2.resize(preview_source, size)

    if focus_mask is not None:
        mask = cv2.resize(focus_mask, size, interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 10]

        if valid_contours:
            contour_points = np.concatenate(valid_contours, axis=0)
            x, y, w_box, h_box = cv2.boundingRect(contour_points)
            pad = max(4, int(max(w_box, h_box) * 0.18))
            pt1 = (max(0, x - pad), max(0, y - pad))
            pt2 = (
                min(size[0] - 1, x + w_box + pad),
                min(size[1] - 1, y + h_box + pad),
            )
            cv2.rectangle(preview, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(
                preview,
                "Tumor",
                (pt1[0], max(12, pt1[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            return preview

    if not yolo_boxes:
        return preview

    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2, conf = max(yolo_boxes, key=lambda box: box[4])
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    dx1, dy1, dx2, dy2 = x1, y1, x2, y2
    scale_x = size[0] / image_rgb.shape[1]
    scale_y = size[1] / image_rgb.shape[0]
    pt1 = (int(dx1 * scale_x), int(dy1 * scale_y))
    pt2 = (int(dx2 * scale_x), int(dy2 * scale_y))
    cv2.rectangle(preview, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(
        preview,
        f"Tumor {conf:.2f}",
        (pt1[0], max(12, pt1[1] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return preview


def build_display_lung_mask(image_rgb, fallback_mask=None):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, body_mask = cv2.threshold(blurred, 12, 255, cv2.THRESH_BINARY)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_region = np.zeros_like(body_mask)
    if contours:
        cv2.drawContours(body_region, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    body_pixels = blurred[body_region > 0]
    if body_pixels.size == 0:
        if fallback_mask is None:
            return np.zeros_like(gray)
        return np.where(fallback_mask > 0, 255, 0).astype(np.uint8)

    threshold_value = int(np.percentile(body_pixels, 35))
    threshold_value = max(45, min(150, threshold_value))

    _, dark_regions = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    dark_regions = cv2.bitwise_and(dark_regions, body_region)
    dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_regions)
    image_area = gray.shape[0] * gray.shape[1]
    selected_labels = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < image_area * 0.02 or area > image_area * 0.30:
            continue
        if x <= 1 or y <= 1 or (x + w) >= gray.shape[1] - 1 or (y + h) >= gray.shape[0] - 1:
            continue
        selected_labels.append((area, label))

    selected_labels.sort(reverse=True)
    lung_mask = np.zeros_like(gray, dtype=np.uint8)
    for _, label in selected_labels[:2]:
        lung_mask[labels == label] = 255

    if np.count_nonzero(lung_mask) == 0 and fallback_mask is not None:
        lung_mask = np.where(fallback_mask > 0, 255, 0).astype(np.uint8)

    if np.count_nonzero(lung_mask) == 0:
        return lung_mask

    lung_mask = cv2.medianBlur(lung_mask, 5)
    lung_contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(lung_mask)
    if lung_contours:
        cv2.drawContours(filled_mask, lung_contours, -1, 255, -1)
        lung_mask = filled_mask

    return lung_mask


def build_lung_segmentation_view(image_rgb, lung_mask):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    display_mask = build_display_lung_mask(image_rgb, fallback_mask=lung_mask)
    lung_view = np.zeros_like(gray)
    lung_view[display_mask > 0] = gray[display_mask > 0]

    return cv2.cvtColor(lung_view, cv2.COLOR_GRAY2RGB)


def resize_with_padding(image_rgb, size, background=(0, 0, 0)):
    target_w, target_h = size

    if image_rgb is None or image_rgb.size == 0:
        return np.full((target_h, target_w, 3), background, dtype=np.uint8)

    height, width = image_rgb.shape[:2]
    scale = min(target_w / max(width, 1), target_h / max(height, 1))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=interpolation)

    canvas = np.full((target_h, target_w, 3), background, dtype=image_rgb.dtype)
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas[offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = resized
    return canvas


def build_zoomed_tumor_only_view(segmented_crop, crop_mask, contours, size=(224, 224), padding_ratio=0.25):
    contour_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(contour_points)

    pad_x = max(10, int(w * padding_ratio))
    pad_y = max(10, int(h * padding_ratio))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(segmented_crop.shape[1], x + w + pad_x)
    y2 = min(segmented_crop.shape[0], y + h + pad_y)

    zoom_crop = segmented_crop[y1:y2, x1:x2].copy()
    zoom_mask = crop_mask[y1:y2, x1:x2]
    zoom_contours, _ = cv2.findContours(zoom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if zoom_contours:
        cv2.drawContours(zoom_crop, zoom_contours, -1, (255, 255, 255), 1)

    return resize_with_padding(zoom_crop, size)


def build_irregular_tumor_views(source_image_rgb, display_image_rgb, yolo_boxes, size=(224, 224)):
    boundary_view = display_image_rgb.copy()
    tumor_only_view = np.zeros((size[1], size[0], 3), dtype=display_image_rgb.dtype)
    focus_mask_view = np.zeros((size[1], size[0]), dtype=np.uint8)

    if not yolo_boxes:
        return cv2.resize(boundary_view, size), tumor_only_view, False, focus_mask_view

    crop_rgb, (x1, y1, x2, y2) = get_best_nodule_crop_and_box(source_image_rgb, yolo_boxes)
    segmented_crop, crop_mask = segment_tumor_irregular(crop_rgb)

    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(boundary_view, size), tumor_only_view, False, focus_mask_view

    tumor_only_view = build_zoomed_tumor_only_view(segmented_crop, crop_mask, contours, size=size)

    full_focus_mask = np.zeros(display_image_rgb.shape[:2], dtype=np.uint8)
    full_focus_mask[y1:y2, x1:x2] = crop_mask
    focus_mask_view = cv2.resize(full_focus_mask, size, interpolation=cv2.INTER_NEAREST)

    for contour in contours:
        shifted = contour + np.array([[[x1, y1]]])
        cv2.drawContours(boundary_view, [shifted], -1, (255, 0, 0), 2)

    return cv2.resize(boundary_view, size), tumor_only_view, True, focus_mask_view


# -------------------------------------------------- 
# # 🚀 MAIN PIPELINE # 
# --------------------------------------------------
def run_pipeline_legacy(image_path):
    yolo_boxes = []

    original = cv2.imread(image_path)

    # Preprocess for segmentation/classifier
    processed_img = preprocess_ct_for_models(image_path)

    # 🫁 Lung segmentation
    lung_for_yolo, _ = apply_lung_mask(processed_img)

    # 🔥 YOLO SPECIAL PREPROCESS (THIS WAS MISSING)
    yolo_input = preprocess_for_yolo(lung_for_yolo)

    # 🎯 Detect nodules
    yolo_boxes = detect_tumor_yolo(original)
    # --------------------------------------------------
    # 3. IF NO NODULE FOUND -> STOP
    # --------------------------------------------------
    if len(yolo_boxes) == 0:
        lung_segmented, _ = apply_lung_mask(processed_img)
        return {
            "original": original,
            "preprocessed": processed_img,
            "lung_segmented": lung_segmented,
            "prediction": "No Nodule Detected",
            "confidence": 0.0,
            "risk": risk_level(0, [0], "No Nodule Detected", [0.0]),
            "nodules": 0,
            "tumor_detected": False
        }

    # --------------------------------------------------
    # 4. CROP DETECTED NODULE FROM ORIGINAL IMAGE
    # --------------------------------------------------
    def crop_nodule_region(original_image, yolo_boxes):
        h, w, _ = original_image.shape

        # -----------------------------
        # If YOLO FAILS → SAFE CENTER CROP
        # -----------------------------
        if len(yolo_boxes) == 0:
            cx, cy = w // 2, h // 2
            size = min(h, w) // 3   # smaller safer crop

            x1 = max(0, cx - size)
            y1 = max(0, cy - size)
            x2 = min(w, cx + size)
            y2 = min(h, cy + size)

            crop = original_image[y1:y2, x1:x2]

        else:
            # -----------------------------
            # Use YOLO BOX
            # -----------------------------
            x1,y1,x2,y2,conf = yolo_boxes[0]

            pad = 30
            x1 = max(0, int(x1 - pad))
            y1 = max(0, int(y1 - pad))
            x2 = min(w, int(x2 + pad))
            y2 = min(h, int(y2 + pad))

            crop = original_image[y1:y2, x1:x2]

        # -----------------------------
        # 🚨 FINAL SAFETY CHECK
        # -----------------------------
        if crop is None or crop.size == 0:
            print("⚠️ Crop failed — using full lung fallback")
            crop = original_image.copy()

        return crop

    # --------------------------------------------------
    # 5. CROP → THEN APPLY LUNG SEGMENTATION
    # --------------------------------------------------

    # ⭐ FIRST crop the detected YOLO nodule
    nodule_crop = crop_nodule_region(original, yolo_boxes)

    # ⭐ THEN apply lung segmentation on the crop
    nodule_crop, _ = apply_lung_mask(nodule_crop)

    # ⭐ Resize for classifier
    nodule_crop = cv2.resize(nodule_crop, (224, 224))

    # Full lung segmentation for visualization
    lung_segmented, _ = apply_lung_mask(processed_img)

    # Full-size segmentation is still kept for visualization/output.
    lung_segmented, _ = apply_lung_mask(processed_img)

    # --------------------------------------------------
    # 6. CLASSIFICATION ON SEGMENTED NODULE CROP
    # --------------------------------------------------
    crop_pil = Image.fromarray(nodule_crop)
    input_tensor = transform(crop_pil).unsqueeze(0).to(device)

    classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        outputs = classifier(input_tensor)

    probs = torch.softmax(outputs, dim=1)
    cancer_prob = float(probs[0][0])
    prediction = "Cancer" if cancer_prob > 0.35 else "Non-Cancer"

    if prediction == "Non-Cancer":
        return {
            "original": original,
            "preprocessed": processed_img,
            "lung_segmented": lung_segmented,
            "prediction": prediction,
            "confidence": cancer_prob,
            "risk": risk_level(0, [0], prediction, [cancer_prob]),
            "nodules": 0,
            "tumor_detected": False
        }

    # --------------------------------------------------
    # 7. GRADCAM ON THE SEGMENTED NODULE CROP
    # --------------------------------------------------
    heatmap, cam = generate_gradcam(input_tensor)
    lung_resized = cv2.resize(lung_segmented, (224, 224))
    overlay_base = build_reference_overlay_base(processed_img)

    # --------------------------------------------------
    # 8. BUILD CANDIDATE TUMOR MAP
    # --------------------------------------------------
    cam_map = (cam * 255).astype("uint8")
    lung_mask = get_lung_mask_224(lung_resized)
    intensity = ct_intensity_map(lung_resized)
    edges = texture_edges(lung_resized)

    candidate = cv2.addWeighted(cam_map, 0.4, intensity, 0.4, 0)
    candidate = cv2.addWeighted(candidate, 0.8, edges, 0.2, 0)
    candidate = cv2.bitwise_and(candidate, lung_mask)
    candidate = cv2.GaussianBlur(candidate, (15, 15), 0)

    # --------------------------------------------------
    # 9. OTSU THRESHOLD + MORPHOLOGY
    # --------------------------------------------------
    _, tumor_mask = cv2.threshold(
        candidate, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((11, 11), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # --------------------------------------------------
    # 10. KEEP ROUND / IRREGULAR BLOBS ONLY
    # --------------------------------------------------
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(tumor_mask)
    valid_contours = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 40:
            continue

        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)

        if 0.15 < circularity < 0.9:
            valid_contours.append(c)
            cv2.drawContours(clean_mask, [c], -1, 255, -1)

    tumor_mask = clean_mask
    nodules = len(valid_contours)

    kernel = np.ones((7, 7), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(tumor_mask)

    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

        if 80 < area < 5000 and circularity > 0.2:
            valid_contours.append(c)
            cv2.drawContours(clean_mask, [c], -1, 255, -1)

    tumor_mask = clean_mask
    nodules = len(valid_contours)

    # --------------------------------------------------
    # 11. BUILD VISUAL OUTPUTS
    # --------------------------------------------------
    tumor_seg = lung_resized.copy()

    if len(valid_contours) > 0:
        cv2.drawContours(tumor_seg, valid_contours, -1, (255, 0, 0), 2)
        tumor_detected_flag = True
    else:
        tumor_detected_flag = False

    heatmap, overlay = build_heatmap_overlay(
        overlay_base,
        cam,
        lung_mask=lung_mask,
        tumor_mask=tumor_mask,
    )
    tumor_color_overlay = build_single_color_tumor_overlay(
        overlay_base,
        tumor_mask,
        size=overlay_base.shape[1::-1],
    )

    return {
        "original": original,
        "preprocessed": processed_img,
        "lung_segmented": lung_segmented,
        "tumor_segmented": tumor_seg,
        "heatmap": heatmap,
        "overlay": overlay,
        "tumor_color_overlay": tumor_color_overlay,
        "prediction": prediction,
        "confidence": cancer_prob,
        "risk": risk_level(nodules, [50], prediction, [cancer_prob]),
        "nodules": nodules,
        "tumor_detected": tumor_detected_flag
    }


def run_pipeline(image_path):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    processed_img = preprocess_ct_for_models(image_path)
    lung_only, lung_mask = apply_lung_mask(processed_img)
    lung_segmented = build_lung_segmentation_view(processed_img, lung_mask)
    yolo_input = preprocess_for_yolo_same_size(lung_only)
    yolo_boxes = detect_tumor_yolo(yolo_input)

    prediction, cancer_prob, input_tensor, lung_resized = classify_ct_slice(lung_only)
    nodule_count = len(yolo_boxes)
    tumor_boundary_view, tumor_only_view, irregular_boundary_found, tumor_focus_mask = build_irregular_tumor_views(
        lung_only,
        lung_segmented,
        yolo_boxes,
    )
    box_tumor_detected = build_detection_preview(
        lung_segmented,
        yolo_boxes,
        focus_mask=tumor_focus_mask,
    )
    overlay_base = build_reference_overlay_base(processed_img)

    base_result = {
        "original": original,
        "preprocessed": processed_img,
        "lung_segmented": lung_segmented,
        "box_tumor_detected": box_tumor_detected,
        "tumor_segmented": tumor_boundary_view,
        "tumor_only_segmented": tumor_only_view,
        "prediction": prediction,
        "confidence": cancer_prob,
        "risk": risk_level(nodule_count, [0], prediction, [cancer_prob]),
        "nodules": nodule_count,
        "tumor_detected": nodule_count > 0 or irregular_boundary_found,
    }

    if prediction != "Cancer":
        return base_result

    heatmap, cam = generate_gradcam(input_tensor)
    cam_map = (cam * 255).astype("uint8")
    lung_mask = get_lung_mask_224(lung_resized)
    intensity = ct_intensity_map(lung_resized)
    edges = texture_edges(lung_resized)

    candidate = cv2.addWeighted(cam_map, 0.4, intensity, 0.4, 0)
    candidate = cv2.addWeighted(candidate, 0.8, edges, 0.2, 0)
    candidate = cv2.bitwise_and(candidate, lung_mask)
    candidate = cv2.GaussianBlur(candidate, (15, 15), 0)

    _, tumor_mask = cv2.threshold(
        candidate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((11, 11), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(tumor_mask)
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)

        if 0.15 < circularity < 0.9:
            valid_contours.append(contour)
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)

    tumor_mask = clean_mask

    kernel = np.ones((7, 7), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(tumor_mask)
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

        if 80 < area < 5000 and circularity > 0.2:
            valid_contours.append(contour)
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)

    tumor_mask = clean_mask
    tumor_seg = tumor_boundary_view.copy()

    if valid_contours:
        cv2.drawContours(tumor_seg, valid_contours, -1, (255, 0, 0), 2)

    heatmap, overlay = build_heatmap_overlay(
        overlay_base,
        cam,
        lung_mask=lung_mask,
        tumor_mask=tumor_mask,
        focus_mask=tumor_focus_mask,
    )
    tumor_color_focus_mask = tumor_focus_mask
    if tumor_color_focus_mask is None or np.count_nonzero(tumor_color_focus_mask) == 0:
        tumor_color_focus_mask = tumor_mask

    tumor_color_overlay = build_single_color_tumor_overlay(
        overlay_base,
        tumor_color_focus_mask,
        size=overlay_base.shape[1::-1],
    )

    base_result.update(
        {
            "tumor_segmented": tumor_seg,
            "heatmap": heatmap,
            "overlay": overlay,
            "tumor_color_overlay": tumor_color_overlay,
            "risk": risk_level(nodule_count, [50], prediction, [cancer_prob]),
            "tumor_detected": nodule_count > 0 or irregular_boundary_found or bool(valid_contours),
        }
    )

    return base_result
