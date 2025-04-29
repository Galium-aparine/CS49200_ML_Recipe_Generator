from ultralytics import YOLO
import cv2
import torch

# Load the YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # 'n' is the smallest model, replace with 's', 'm', 'l' for more accuracy

# Check if CUDA (GPU) is available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model on the Open Images V7 dataset
results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)
