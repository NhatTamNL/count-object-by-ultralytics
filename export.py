from ultralytics import YOLO

model = YOLO("sendo.pt")

model.export(format='onnx', opset=11)