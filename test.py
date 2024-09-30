from ultralytics import YOLO

videoPath = 'coffee2_49.mp4'
# cap = cv2.VideoCapture(videoPath)

model = YOLO('yolov8n.pt')

model.track(videoPath, show=True, classes=[0], conf=0.6, save=True)