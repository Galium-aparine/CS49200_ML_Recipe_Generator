from ultralytics import YOLO
import cv2
import torch

# Load the YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # 'n' is the smallest model, replace with 's', 'm', 'l' for more accuracy

# Check if CUDA (GPU) is available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Run inference on an image
results = model("image_recog/fridge.jpg") #, save=True)  # Replace with your image file

# Extract detection results
detections = []
for result in results:
    for box in result.boxes:
        detections.append({
            "class_id": int(box.cls.item()),  # Class index
            "class_name": model.names[int(box.cls.item())],  # Class name
            # "confidence": float(box.conf.item()),  # Confidence score
            # "bbox": box.xyxy.tolist()[0]  # Bounding box (x1, y1, x2, y2)
        })

# Print the list of detected items
print(detections)
