# import cv2
# from ultralytics import YOLO
# from collections import defaultdict

# model = YOLO("Tam.pt")

# video_path = "video_test/2.mp4"
# cap = cv2.VideoCapture(video_path)

# class_counts = defaultdict(int)
# seen_ids = set()

# filetxt = "object_counts1.txt"

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         results = model.track(frame, persist=True)
        
#         for result in results:
#             boxes = result.boxes
            
#             if boxes.id is not None:
#                 track_ids = boxes.id.cpu().numpy()
#                 class_ids = boxes.cls.cpu().numpy()
                
#                 for track_id, class_id in zip(track_ids, class_ids):
#                     if int(track_id) not in seen_ids:
#                         seen_ids.add(int(track_id))
#                         class_counts[int(class_id)] += 1
#                     else:
#                         print(f"Tracking ID {track_id} has already been counted.")
#             else:
#                 print("Tracking is not enabled for these boxes.") 
        
#         annotated_frame = results[0].plot()
#         # cv2.imshow("YOLOv8 Tracking", annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()

# with open(filetxt, "w") as f:
#     for class_id, count in class_counts.items():
#         f.write(f"Class ID: {class_id}, Count: {count}\n")

# print(f"Object counts have been saved to {filetxt}")


# import cv2
# from ultralytics import YOLO
# from collections import defaultdict

# model = YOLO("Tam.pt")


# class_counts = defaultdict(int)
# seen_ids = set()

# filetxt = "object_counts1.txt"

# frame = cv2.imread("1.jpg")

# results = model.track(frame, persist=True, show=True)
    
# for result in results:
#     boxes = result.boxes
    
#     if boxes.id is not None:
#         track_ids = boxes.id.cpu().numpy()
#         class_ids = boxes.cls.cpu().numpy()
        
#         for track_id, class_id in zip(track_ids, class_ids):
#             if int(track_id) not in seen_ids:
#                 seen_ids.add(int(track_id))
#                 class_counts[int(class_id)] += 1
#             else:
#                 print(f"Tracking ID {track_id} has already been counted.")
#     else:
#         print("Tracking is not enabled for these boxes.") 

# annotated_frame = results[0].plot()
# # cv2.imshow("YOLOv8 Tracking", annotated_frame)

# # cap.release()
# cv2.destroyAllWindows()

# with open(filetxt, "w") as f:
#     for class_id, count in class_counts.items():
#         f.write(f"Class ID: {class_id}, Count: {count}\n")

# print(f"Object counts have been saved to {filetxt}")



import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("Tam.pt")
filetxt = "object_counts1.txt"

video_path = "video_test/2.mp4"
cap = cv2.VideoCapture(video_path)

seen_ids = set()
class_counts = defaultdict(int)
track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.cpu().numpy()
        annotated_frame = results[0].plot()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            
            if track_id not in seen_ids:
                seen_ids.add(track_id)
                class_counts[int(class_id)] += 1

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

with open(filetxt, "w") as f:
    for class_id, count in class_counts.items():
        f.write(f"Class ID: {class_id}, Count: {count}\n")

print(f"Object counts have been saved to {filetxt}")