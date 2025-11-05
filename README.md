# PIMPLE REMOVER TOOL

## Project Flow
YOLOv8s Training (On ACNE04) dataset --> Processing bounding boxes as binary masks for inpainting --> Using Deepfillv2 for inpainting 
(--> Using SAM based Techniques for pixel wise segmentation)

## Project Outcomes
- **Confidence = 0.3, Iterations = 1, Kernel Size = (3,3)** ![WhatsApp Image 2025-11-05 at 13 28 02_79715e09](https://github.com/user-attachments/assets/9bd581e6-3d05-4839-bc40-652aa6be7940)
- **Confidence = 0.1, Iterations = 2, Kernel Size = (5,5)** ![WhatsApp Image 2025-11-05 at 13 30 24_e1d89364](https://github.com/user-attachments/assets/8488a69d-fcca-4881-90a2-5fdc8951bb95)
- **Confidence = 0.05, Iterations = 3, Kernel Size = (7,7)** ![WhatsApp Image 2025-11-05 at 13 32 02_1b7f7639](https://github.com/user-attachments/assets/75625923-4088-4513-9e01-716c000bcd1e)




## Training:
- training YOLOv8s on 1257 training images, for 150 epochs, with the best model achieving a mAP50 of 0.632.
- using dataset with lesser severe acnes for inference
- Data.yaml
  ```json
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
  


## Inpainting:
