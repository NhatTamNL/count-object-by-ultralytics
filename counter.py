import cv2
from ultralytics import YOLO, solutions
from ultralytics.utils.plotting import *

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture("2.mp4")
assert cap.isOpened(), "Error reading video file"

classes_to_count = [0]


# counter = solutions.ObjectCounter(view_img=True, names=model.names)
counter_track = solutions.CounterTracker(names=model.names)
 
def extract_and_process_tracks(im0, tracks):
    # Annotator Init and region drawing
    annotator = Annotator(im0, 2, model.names)

    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            annotator.box_label(box, label=f"{model.names[0]}#{track_id}", color=colors(int(track_id), True))
    return im0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

    # im0 = counter.start_counting(im0, tracks)
    # im0 = counter_track.start_counting(im0, tracks)
    im0 = extract_and_process_tracks(im0, tracks)
    
    for result in tracks:
        boxes = result.boxes
        if boxes.is_track:
            track_ids = boxes.id
            print(f"Tracking IDs: {track_ids} total {len(track_ids)}")
        else:
            print("Tracking is not enabled for these boxes.")
            
    cv2.imshow("Counter", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
               
cap.release()
cv2.destroyAllWindows()



