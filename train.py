from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  

# Train the model
results = model.train(data="sendo/data.yaml", epochs=20, imgsz=640)