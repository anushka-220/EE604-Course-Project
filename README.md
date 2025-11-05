# PIMPLE REMOVER TOOL

## Project Flow
YOLOv8s Training (On ACNE04) dataset --> Processing bounding boxes as binary masks for inpainting --> Using Deepfillv2 for inpainting 
(--> Using SAM based Techniques for pixel wise segmentation)

## Project Outcomes





## Training:
- training YOLOv8s on 1257 training images, for 150 epochs, with the best model achieving a mAP50 of 0.632.
- using dataset with lesser severe acnes for inference
- Data.yaml
  ```python
    # --- Create the data.yaml file ---

  class_names = [
      'pimple'
  ] # e.g., ['pimple', 'blackhead', 'whitehead']
  
  yaml_content = f"""
  path: {output_dir}  # dataset root directory
  train: images/train  # train images (relative to 'path')
  val: images/val      # val images (relative to 'path')
  test: images/test    # test images (relative to 'path')
  
  # Classes
  names:
    {os.linesep.join([f'  {i}: {name}' for i, name in enumerate(class_names)])}
  """
  
  yaml_path = os.path.join(output_dir, 'data.yaml')
  with open(yaml_path, 'w') as f:
      f.write(yaml_content)
  
  print(f"Created data.yaml at {yaml_path}")
  ```
- Code:
  ```python
  # ============================================
  # 🚀 YOLOv8 Model Training Script (Acne Dataset with Augmentation)
  # ============================================
  
  # 1️⃣ Install dependencies
  !pip install -q ultralytics
  
  # 2️⃣ Import library
  from ultralytics import YOLO
  import torch
  
  # 3️⃣ Auto-select device
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"✅ Using device: {device.upper()}")
  
  # 4️⃣ Load pretrained model
  # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
  model = YOLO('yolov8s.pt')
  
  # 5️⃣ Define dataset YAML path
  data_yaml = '/kaggle/working/acne_dataset/data.yaml'
  
  # 6️⃣ Start training with augmentations
  print("\n🔧 Starting YOLOv8 training with augmentations...")
  results = model.train(
      data=data_yaml,              
      epochs=150,                  
      imgsz=640,                   
      batch=16,                    
      name='yolov8s_acne_augmented', 
      device=device,               
      optimizer='AdamW',           
      patience=20,                 
      verbose=True,                
  
      # ⚡ Augmentation hyperparameters
      degrees=10.0,          # random rotation
      translate=0.1,         # image translation
      scale=0.5,             # random scaling
      shear=0.0,             # shear transform
      perspective=0.0,       # perspective distortion
      flipud=0.0,            # vertical flip (usually off for face data)
      fliplr=0.5,            # horizontal flip
      mosaic=1.0,            # use mosaic augmentation
      mixup=0.1,             # mixup augmentation
      erasing=0.4,           # random erasing (helps generalization)
      hsv_h=0.015,           # hue variation
      hsv_s=0.7,             # saturation variation
      hsv_v=0.4,             # brightness variation
  )
  
  # 7️⃣ Print results summary
  print("\n✅ Training complete!")
  print(f"📁 Results saved at: {results.save_dir}")
  ```
  
## Inpainting and Inference:
- Batch Inference:
  ```python
    # ============================================
  # 🧰 STEP 0 — Setup environment and imports
  # ============================================
  !pip install ultralytics gdown tqdm -q
  !git clone https://github.com/vrindaprabhu/deepfillv2_colab.git
  
  import os, cv2, torch, glob, numpy as np, shutil, subprocess, time
  from ultralytics import YOLO
  from tqdm import tqdm
  import matplotlib.pyplot as plt
  
  # Force GPU usage if available
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"🚀 Using device: {device}")
  
  # ============================================
  # ⚙️ STEP 1 — Load YOLO acne detection model
  # ============================================
  yolo_model_path = "/content/best_final.pt"   # your trained acne model
  model = YOLO(yolo_model_path)
  model.to(device)
  
  # Directory setup
  os.makedirs("/content/images", exist_ok=True)
  os.makedirs("/content/masked", exist_ok=True)
  os.makedirs("/content/cleaned", exist_ok=True)
  
  # ============================================
  # 🧩 STEP 2 — Run YOLO & create acne masks
  # ============================================
  image_files = glob.glob("/content/images/*.jpg") + glob.glob("/content/images/*.png")
  print(f"Found {len(image_files)} input images")
  
  for img_path in tqdm(image_files, desc="YOLO Detection", unit="img"):
      img = cv2.imread(img_path)
      H, W, _ = img.shape
      results = model.predict(img, conf=0.3, device=device, verbose=False)[0]
  
      mask = np.zeros((H, W), dtype=np.uint8)
      for box in results.boxes.xyxy:
          x1, y1, x2, y2 = map(int, box)
          cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
  
      # Optional dilation for smoother inpainting
      kernel = np.ones((5,5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=1)
  
      base = os.path.splitext(os.path.basename(img_path))[0]
      cv2.imwrite(f"/content/masked/{base}_input.png", img)
      cv2.imwrite(f"/content/masked/{base}_mask.png", mask)
  
  print("✅ YOLO completed — masks saved in /content/masked")
  
  # ============================================
  # 🎨 STEP 3 — Setup DeepFill v2 model
  # ============================================
  !gdown "https://drive.google.com/u/0/uc?id=1uMghKl883-9hDLhSiI8lRbHCzCmmRwV-&export=download"
  !mv deepfillv2_WGAN_G_epoch40_batchsize4.pth deepfillv2_colab/model/deepfillv2_WGAN.pth
  
  # ============================================
  # 🧠 STEP 4 — Run DeepFill v2 via CLI for each image
  # ============================================
  input_dir = "/content/masked"
  output_dir = "/content/cleaned"
  os.makedirs(output_dir, exist_ok=True)
  
  pairs = [f for f in os.listdir(input_dir) if f.endswith("_input.png")]
  timings = []
  
  for file in tqdm(pairs, desc="DeepFill v2 Cleaning", unit="img"):
      base = file.replace("_input.png", "")
      img_path = os.path.join(input_dir, f"{base}_input.png")
      mask_path = os.path.join(input_dir, f"{base}_mask.png")
  
      # Copy to DeepFill input folder
      shutil.copy(img_path, "/content/deepfillv2_colab/input/input_img.png")
      shutil.copy(mask_path, "/content/deepfillv2_colab/input/mask.png")
  
      t0 = time.perf_counter()
      subprocess.run(["python", "inpaint.py"], cwd="/content/deepfillv2_colab", check=True)
      t1 = time.perf_counter()
      timings.append(t1 - t0)
  
      shutil.move("/content/deepfillv2_colab/output/inpainted_img.png",
                  f"{output_dir}/{base}_cleaned.png")
  
  print(f"✅ DeepFill completed — cleaned images in /content/cleaned")
  print(f"⏱️ Time per image — min:{min(timings):.2f}s  median:{np.median(timings):.2f}s  max:{max(timings):.2f}s")
  
  # ============================================
  # 🖼️ STEP 5 — Display sample result
  # ============================================
  sample = os.listdir("/content/cleaned")[0]
  img1 = cv2.cvtColor(cv2.imread(f"/content/masked/{sample.replace('_cleaned.png','_input.png')}"), cv2.COLOR_BGR2RGB)
  img2 = cv2.cvtColor(cv2.imread(f"/content/cleaned/{sample}"), cv2.COLOR_BGR2RGB)
  
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1); plt.imshow(img1); plt.title("Original Image"); plt.axis("off")
  plt.subplot(1,2,2); plt.imshow(img2); plt.title("Acne Removed (DeepFill v2)"); plt.axis("off")
  plt.show()
  ```
- Inference for a single image:
  ```python
  # ======================================================
  # 💡 Lightweight version (no re-clone / re-install)
  # ======================================================
  import os, cv2, torch, shutil, subprocess, time, numpy as np
  import matplotlib.pyplot as plt
  from google.colab import files
  from ultralytics import YOLO
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"🚀 Using device: {device}")
  
  # Load YOLO model (already trained)
  model = YOLO("/content/best_final.pt").to(device)
  
  # Create directories if missing
  os.makedirs("/content/images", exist_ok=True)
  os.makedirs("/content/masked", exist_ok=True)
  os.makedirs("/content/cleaned", exist_ok=True)
  
  # ======================================================
  # 📤 STEP 1 — Upload a face image
  # ======================================================
  print("📤 Please upload a face image to clean acne:")
  uploaded = files.upload()
  
  # Check if any files were uploaded
  if not uploaded:
      print("No files were uploaded. Please try again.")
  else:
      # Get the list of uploaded file names
      uploaded_files = list(uploaded.keys())
  
      # Print the files in the images directory for debugging
      print("Files in /content/images:", os.listdir("/content/images"))
  
      # Assuming the first uploaded file is the one to process
      img_name = uploaded_files[0]
      img_path = f"/content/images/{img_name}"
  
      # Save the uploaded file to the images directory
      with open(img_path, 'wb') as f:
          f.write(uploaded[img_name])
  
      print(f"✅ Uploaded and saved: {img_name}")
  
      # ======================================================
      # 🧩 STEP 2 — Run YOLO detection and create mask
      # ======================================================
      img = cv2.imread(img_path)
  
      # Check if the image was loaded successfully
      if img is None:
          print(f"Error: Could not load image from {img_path}. Please check the file path and format.")
      else:
          H, W, _ = img.shape
          results = model.predict(img, conf=0.05, device=device, verbose=False)[0]
  
          mask = np.zeros((H, W), dtype=np.uint8)
          for box in results.boxes.xyxy:
              x1, y1, x2, y2 = map(int, box)
              cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
  
          # Optional: slightly dilate for smoother fill
          kernel = np.ones((7,7), np.uint8)
          mask = cv2.dilate(mask, kernel, iterations=3)
  
          base = os.path.splitext(img_name)[0]
          masked_img = f"/content/masked/{base}_input.png"
          masked_mask = f"/content/masked/{base}_mask.png"
  
          cv2.imwrite(masked_img, img)
          cv2.imwrite(masked_mask, mask)
          print("✅ Mask created successfully")
  
          # ======================================================
          # 🧠 STEP 3 — Run DeepFill v2 inpainting
          # ======================================================
          shutil.copy(masked_img, "/content/deepfillv2_colab/input/input_img.png")
          shutil.copy(masked_mask, "/content/deepfillv2_colab/input/mask.png")
  
          print("🎨 Running DeepFill v2 (this may take a few seconds)...")
          start = time.perf_counter()
          subprocess.run(["python", "inpaint.py"], cwd="/content/deepfillv2_colab", check=True)
          end = time.perf_counter()
  
          cleaned_img = f"/content/cleaned/{base}_cleaned.png"
          shutil.move("/content/deepfillv2_colab/output/inpainted_img.png", cleaned_img)
          print(f"✅ Done! (inpainting time: {end-start:.2f}s)")
  
          # ======================================================
          # 🖼️ STEP 4 — Display before vs after
          # ======================================================
          img_before = cv2.cvtColor(cv2.imread(masked_img), cv2.COLOR_BGR2RGB)
          img_after  = cv2.cvtColor(cv2.imread(cleaned_img), cv2.COLOR_BGR2RGB)
  
          plt.figure(figsize=(10,5))
          plt.subplot(1,2,1); plt.imshow(img_before); plt.title("Before (Original)"); plt.axis("off")
          plt.subplot(1,2,2); plt.imshow(img_after);  plt.title("After (Acne Removed)"); plt.axis("off")
          plt.show()
  ```
- Control Parameters and Outcomes:
  - **Confidence = 0.3, Iterations = 1, Kernel Size = (3,3)** ![WhatsApp Image 2025-11-05 at 13 28 02_79715e09](https://github.com/user-attachments/assets/9bd581e6-3d05-4839-bc40-652aa6be7940)
- **Confidence = 0.1, Iterations = 2, Kernel Size = (5,5)** ![WhatsApp Image 2025-11-05 at 13 30 24_e1d89364](https://github.com/user-attachments/assets/8488a69d-fcca-4881-90a2-5fdc8951bb95)
- **Confidence = 0.05, Iterations = 3, Kernel Size = (7,7)** ![WhatsApp Image 2025-11-05 at 13 32 02_1b7f7639](https://github.com/user-attachments/assets/75625923-4088-4513-9e01-716c000bcd1e)

## Example usage of modular code
# ======================================================
# 💡 Example Usage
# ======================================================
  ```python
  # Example: Acne Removal Pipeline
  
  from google.colab import files
  
  # 1️⃣ Upload a face image
  uploaded = files.upload()
  img_path = list(uploaded.keys())[0]
  img_full_path = f"/content/images/{img_path}"
  
  # 2️⃣ Detect acne
  img, mask, boxes = detect_acne(
      image_path=img_full_path,
      model_path="/content/best_final.pt",
      conf=0.1,        # detection confidence
      pad=15           # padding around acne boxes
  )
  
  # 3️⃣ Cleanse skin
  cleaned = cleanse_skin(
      img=img,
      mask=mask,
      kernel_size=7,   # mask dilation kernel size
      iterations=3     # mask dilation iterations
  )
  
  # 4️⃣ Visualize results
  img_before = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_after = cv2.cvtColor(cv2.imread(cleaned), cv2.COLOR_BGR2RGB)
  
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1); plt.imshow(img_before); plt.title("Before"); plt.axis("off")
  plt.subplot(1,2,2); plt.imshow(img_after);  plt.title("After"); plt.axis("off")
  plt.show()
  ```
