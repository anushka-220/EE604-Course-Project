# PIMPLE REMOVER TOOL

## Project Flow
YOLOv8s Training (On ACNE04) dataset --> Processing bounding boxes as binary masks for inpainting --> Using Deepfillv2 for inpainting 
(--> Using SAM based Techniques for pixel wise segmentation)

## Training:
- training YOLOv8s on 1257 training images, for 150 epochs, with the best model achieving a mAP50 of 0.632.
- using dataset with lesser severe acnes for inference

## Inpainting:
