import os
from ultralytics import YOLO

# Load model gốc
model = YOLO("yolo11n.pt")

# Train với file cấu hình dataset data.yaml
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data.yaml")
results = model.train(data=data_path, epochs=100, imgsz=640)