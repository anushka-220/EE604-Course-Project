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

## Inpainting:
