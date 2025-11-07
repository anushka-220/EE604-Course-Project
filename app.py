import os
import sys
import subprocess
import shutil
import time
import numpy as np
from PIL import Image


# ==============================
# 1️⃣ SETUP ENVIRONMENT
# ==============================
def setup_environment():
    """
    Installs dependencies and sets up DeepFillv2.
    """
    print("--- Setting up environment ---")

    print("[1/3] Installing Python dependencies...")
    # This is the fix
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "gradio", "ultralytics", "gdown", "tqdm", "opencv-python-headless", "scikit-image"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

    print("[2/3] Checking for DeepFillv2 repository...")
    if not os.path.exists('deepfillv2_colab'):
        print("   Cloning DeepFillv2 repository...")
        subprocess.run(["git", "clone", "https://github.com/vrindaprabhu/deepfillv2_colab.git"], check=True)
    else:
        print("   DeepFillv2 repository already exists.")

    print("[3/3] Checking for DeepFillv2 model weights...")
    deepfill_model_path = 'deepfillv2_colab/model/deepfillv2_WGAN.pth'
    gdown_file = 'deepfillv2_WGAN_G_epoch40_batchsize4.pth'
    download_url = "https://drive.google.com/uc?id=1uMghKl883-9hDLhSiI8lRbHCzCmmRwV-"

    if not os.path.exists(deepfill_model_path):
        print("   Downloading DeepFillv2 model weights...")
        try:
            # This is the fix
            subprocess.run([sys.executable, "-m", "gdown", download_url], check=True)
            if os.path.exists(gdown_file):
                shutil.move(gdown_file, deepfill_model_path)
                print("   ✅ Model weights ready.")
            else:
                print("❌ Download failed. Please place manually:")
                print("   deepfillv2_colab/model/deepfillv2_WGAN.pth")
                sys.exit(1)
        except Exception as e:
            print(f"Error downloading DeepFillv2 model: {e}")
            sys.exit(1)
    else:
        print("   DeepFillv2 model weights already exist.")

    print("--- Environment setup complete ---")


# ==============================
# 2️⃣ IMPORTS
# ==============================
try:
    import gradio as gr
    import cv2
    import torch
    from ultralytics import YOLO
except ImportError as e:
    print(f"❌ Missing library: {e}")
    sys.exit(1)


# ==============================
# 3️⃣ GLOBAL CONFIG
# ==============================
YOLO_MODEL_PATH = "G:\\Downloads\\best_final.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE.upper()}")


# ==============================
# 4️⃣ ACNE REMOVAL PIPELINE
# ==============================
def remove_acne(input_image_rgb, conf_threshold, kernel_size_px, dilate_iterations):
    """
    Main acne removal function using YOLO + DeepFillv2.
    """
    if isinstance(input_image_rgb, str):
        input_image_rgb = np.array(Image.open(input_image_rgb).convert("RGB"))
    img_bgr = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2BGR)
    H, W, _ = img_bgr.shape

    base_name = f"temp_{time.time_ns()}"
    input_img_path = f"{base_name}_input.png"
    input_mask_path = f"{base_name}_mask.png"
    final_output_path = f"{base_name}_cleaned.png"

    deepfill_input_img = "deepfillv2_colab/input/input_img.png"
    deepfill_input_mask = "deepfillv2_colab/input/mask.png"
    deepfill_output_img = "deepfillv2_colab/output/inpainted_img.png"

    try:
        print(f"Running YOLO detection (conf={conf_threshold})...")
        model = YOLO(YOLO_MODEL_PATH).to(DEVICE)
        results = model.predict(img_bgr, conf=conf_threshold, device=DEVICE, verbose=False)[0]

        if len(results.boxes) == 0:
            print("⚠️ No detections found. Returning original image.")
            return input_image_rgb

        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Dilate mask
        kernel_val = int(kernel_size_px) // 2 * 2 + 1
        kernel = np.ones((kernel_val, kernel_val), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=int(dilate_iterations))
        print(f"Mask ready | Kernel={kernel_val} | Iter={dilate_iterations}")

        # Save input + mask
        cv2.imwrite(input_img_path, img_bgr)
        cv2.imwrite(input_mask_path, dilated_mask)
        shutil.copy(input_img_path, deepfill_input_img)
        shutil.copy(input_mask_path, deepfill_input_mask)

        if not os.path.exists(deepfill_input_img) or not os.path.exists(deepfill_input_mask):
            raise Exception("DeepFill inputs missing!")

        # Run DeepFillv2
        print("🧩 Running DeepFillv2 inpainting...")
        subprocess.run([sys.executable, "inpaint.py"], cwd="deepfillv2_colab", check=True)

        if not os.path.exists(deepfill_output_img):
            raise Exception("DeepFill output missing — likely model path issue.")

        # Load output
        shutil.move(deepfill_output_img, final_output_path)
        cleaned_img_bgr = cv2.imread(final_output_path)
        cleaned_img_rgb = cv2.cvtColor(cleaned_img_bgr, cv2.COLOR_BGR2RGB)
        print("✅ Inpainting done successfully.")
        return cleaned_img_rgb

    except Exception as e:
        print(f"❌ Error: {e}")
        return input_image_rgb
    finally:
        for f in [input_img_path, input_mask_path, final_output_path]:
            if os.path.exists(f):
                os.remove(f)


# ==============================
# 5️⃣ GRADIO INTERFACE
# ==============================
def create_gradio_app():
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown("# ✨ Pimple Remover Tool (YOLOv8s + DeepFillv2)")
        gr.Markdown("Upload a face image and click **Remove Pimples**.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="Input Face Image")
                conf_slider = gr.Slider(0.05, 0.5, value=0.1, step=0.01, label="Confidence Threshold")
                kernel_slider = gr.Slider(3, 15, value=9, step=2, label="Mask Dilation Kernel Size (px)")
                iter_slider = gr.Slider(1, 5, value=3, step=1, label="Mask Dilation Iterations")
                submit_btn = gr.Button("Remove Pimples", variant="primary")
            with gr.Column(scale=1):
                image_output = gr.Image(type="numpy", label="Cleaned Image")

        submit_btn.click(fn=remove_acne,
                         inputs=[image_input, conf_slider, kernel_slider, iter_slider],
                         outputs=image_output)

        gr.Markdown("---")
        gr.Markdown("""
        ### Recommended Settings
        - **High Quality:** Confidence=0.05, Kernel=3, Iter=2  
        - **Balanced:** Confidence=0.1, Kernel=5, Iter=2
        - **Fast:** Confidence=0.3, Kernel=3, Iter=1
        """)
    return iface


# ==============================
# 6️⃣ MAIN
# ==============================
if __name__ == "__main__":
    setup_environment()

    if not os.path.exists(YOLO_MODEL_PATH):
        print("="*60)
        print(f"❌ YOLO model not found at: {YOLO_MODEL_PATH}")
        print("="*60)
        sys.exit(1)

    app = create_gradio_app()
    print("\n--- Launching Gradio App ---")
    print("Access at http://127.0.0.1:7860 or the public link below.")
    app.launch(debug=True, share=True)
