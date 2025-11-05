import os, cv2, torch, shutil, subprocess, time, numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ======================================================
# 🎨 Helper: Skin Cleansing (Inpainting) Function
# ======================================================
def cleanse_skin(img, mask, output_name="cleaned_face.png",
                 kernel_size=7, iterations=3):
    """
    Cleanse acne regions using DeepFill v2 inpainting.
    
    Args:
        img (ndarray): Original face image (BGR).
        mask (ndarray): Binary acne mask (0/255).
        output_name (str): Filename for output cleaned image.
        kernel_size (int): Size of dilation kernel.
        iterations (int): Number of dilation iterations.
    
    Returns:
        cleaned_path (str): Path to inpainted (acne-free) image.
    """
    # Ensure directories exist
    os.makedirs("/content/masked", exist_ok=True)
    os.makedirs("/content/cleaned", exist_ok=True)

    # Dilation for smoother blending
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=iterations)

    masked_img_path = "/content/masked/input_img.png"
    masked_mask_path = "/content/masked/mask.png"

    cv2.imwrite(masked_img_path, img)
    cv2.imwrite(masked_mask_path, mask)
    print("🎭 Mask refined and saved")

    # Copy to DeepFill directory
    shutil.copy(masked_img_path, "/content/deepfillv2_colab/input/input_img.png")
    shutil.copy(masked_mask_path, "/content/deepfillv2_colab/input/mask.png")

    print("🧠 Running DeepFill v2...")
    start = time.perf_counter()
    subprocess.run(["python", "inpaint.py"], cwd="/content/deepfillv2_colab", check=True)
    end = time.perf_counter()

    cleaned_path = f"/content/cleaned/{output_name}"
    shutil.move("/content/deepfillv2_colab/output/inpainted_img.png", cleaned_path)

    print(f"✅ Inpainting complete in {end-start:.2f}s → {cleaned_path}")
    return cleaned_path
