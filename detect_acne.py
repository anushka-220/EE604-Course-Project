import os, cv2, torch, shutil, subprocess, time, numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ======================================================
# ⚙️ Helper: Acne Detection Function
# ======================================================
def detect_acne(image_path, model_path, conf=0.05, pad=10):
    """
    Detect acne regions using YOLO and return padded bounding boxes & mask.
    
    Args:
        image_path (str): Path to input face image.
        model_path (str): Path to trained YOLO model (.pt).
        conf (float): Confidence threshold for detection.
        pad (int): Padding (in pixels) around bounding boxes for better coverage.
    
    Returns:
        img (ndarray): Loaded image (BGR).
        mask (ndarray): Binary acne mask (0 or 255).
        boxes (list): List of padded bounding boxes [(x1, y1, x2, y2), ...].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Could not load image from {image_path}")
    
    H, W, _ = img.shape
    results = model.predict(img, conf=conf, device=device, verbose=False)[0]

    mask = np.zeros((H, W), dtype=np.uint8)
    boxes = []

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        # Apply padding while clamping to image boundaries
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad)
        y2 = min(H, y2 + pad)
        boxes.append((x1, y1, x2, y2))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    print(f"✅ Detected {len(boxes)} acne regions with conf>{conf}")
    return img, mask, boxes
