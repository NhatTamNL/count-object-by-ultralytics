from ultralytics import YOLO
import os
import random
from object_tracking_counting import YOLOv8ObjectCounter


vidResultsPath = './out'

# if not os.path.isdir(vidResultsPath):
#     os.makedirs(vidResultsPath)
    
# yolo_names = ['yolov8n.pt', 'yolov8m.pt', 'yolov8s.pt',  'yolov8l.pt']
yolo_names = ['yolov8n.pt']
# classes_names = {0: "person", 1: "car"}

colors = []
for _ in range(80):
    rand_tuple = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    colors.append(rand_tuple)
    
counters = []
for yolo_name in yolo_names:
    counter = YOLOv8ObjectCounter(yolo_name, conf = 0.60, classes = [0])
    counters.append(counter)
    
for counter in counters:
    counter.predictVideo(videoPath= '2.mp4', saveDir = vidResultsPath, saveFormat = "avi", display = 'custom', colors = colors)