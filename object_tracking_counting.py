import os
import cv2
import numpy as np
import time
import sort

from ultralytics import YOLO
    
class YOLOv8Detector:
    
    def __init__(self, modelFile = "", label = None, classes = None, conf = 0.25, iou = 0.45):
        self.classes = classes
        self.conf = conf
        self.iou = iou
        
        self.model = YOLO(modelFile)
        self.modelName = modelFile.split('.')[0]
        self.results = None
        
        if label == None:
            self.label = self.model.names
        else:
            self.label = label
            
        
    def predictImg(self, img, verbose=True):
        results = self.model(img, classes = self.classes, conf=self.conf, iou=self.iou, verbose=verbose)
        
        self.origImg = img
        self.results = results[0]
        
        return results[0]
    
    def defaultDisplay(self, show_conf=True, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        if self.results is None:
            raise ValueError('No detected object to display')
        
        displayImg = self.results.plot(show_conf, line_width, font_size, font, pil, example)
        return displayImg
    
    def customDisplay(self, colors, show_cls = True, show_conf = True):
        img = self.origImg
        bbx_thickness = (img.shape[0] + img.shape[1]) // 450

        for box in self.results.boxes:
            textString = ""

            score = box.conf.item() * 100
            class_id = int(box.cls.item())

            # x1 , y1 , x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)
            x1, y1, x2, y2 = np.squeeze(box.xyxy.cpu().numpy()).astype(int)
            

            if show_cls:
                textString += f"{self.label[class_id]}"

            if show_conf:
                textString += f" {score:,.2f}%"

            font = cv2.FONT_HERSHEY_COMPLEX
            fontScale = (((x2 - x1) / img.shape[0]) + ((y2 - y1) / img.shape[1])) / 2 * 2.5
            fontThickness = 1
            textSize, baseline = cv2.getTextSize(textString, font, fontScale, fontThickness)

            img = cv2.rectangle(img, (x1,y1), (x2,y2), colors[class_id], bbx_thickness)
            center_coordinates = ((x1 + x2)//2, (y1 + y2) // 2)

            img =  cv2.circle(img, center_coordinates, 5 , (0,0,255), -1)
            
            if textString != "":
                if (y1 < textSize[1]):
                    y1 = y1 + textSize[1]
                else:
                    y1 -= 2
                img = cv2.rectangle(img, (x1, y1), (x1 + textSize[0] , y1 -  textSize[1]), colors[class_id], cv2.FILLED)
                img = cv2.putText(img, textString, (x1,y1), font, fontScale, (0, 0, 0), fontThickness, cv2.LINE_AA)
        return img
    
    def predictVideo(self, videoPath, saveDir, saveFormat = 'avi', display='custom', verbose=True, **display_args):
        cap = cv2.VideoCapture(videoPath)
        
        vidName = os.path.basename(videoPath)
        width = int(cap.get(3))
        height = int(cap.get(4))
        
        if not os.path.isdir(videoPath):
            os.makedirs(videoPath)
        saveName = self.modelName + '--' + vidName.split('.')[0] + '.' + saveFormat
        saveFile = os.path.join(saveDir, saveName)
        
        if verbose:
            print("----------------------------")
            print(f"DETECTING OBJECTS IN : {vidName} : ")
            print(f"RESOLUTION : {width}x{height}")
            print('SAVING TO :' + saveFile)
        
        out = cv2.VideoWriter(saveFile, cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))

        if not cap.isOpened():
            print("Error opening video stream or file")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading frame")
                break
            
            beg = time.time()
            
            results = self.predictImg(frame, verbose=False)
            if results is None:
                print('***********************************************')
            fps = 1 / (time.time() - beg)
            
            if display == 'default':
                frame = self.defaultDisplay(**display_args)
            elif display == 'custom':
                frame == self.customDisplay(**display_args)

            frame = cv2.putText(frame, f"FPS : {fps:,.2f}", (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        
class YOLOv8ObjectCounter(YOLOv8Detector):
    def __init__(self, modelFile = 'yolov8n.pt', labels= None, classes = None, conf = 0.25, iou = 0.45, 
                trackMaxAge = 45, trackMinHits= 15, trackIouThreshold = 0.3 ):
        
        super().__init__(modelFile , labels, classes, conf, iou)
        
        self.trackMaxAge = trackMaxAge
        self.trackMinHits = trackMinHits
        self.trackIouThreshold = trackIouThreshold
        
    def predictVideo(self, videoPath, saveDir, saveFormat = 'avi', display='custom', verbose=True, **display_args):
        cap = cv2.VideoCapture(videoPath)
        
        vidName = os.path.basename(videoPath)
        width = int(cap.get(3))
        height = int(cap.get(4))
        
        # if not os.path.isdir(videoPath):
            # os.makedirs(videoPath)
            
        saveName = self.modelName + '-' + vidName.split('.')[0] + '.' + saveFormat
        saveFile = os.path.join(saveDir, saveName)
        
        if verbose:
            print("----------------------------")
            print(f"DETECTING OBJECTS IN : {vidName} : ")
            print(f"RESOLUTION : {width}x{height}")
            print('SAVING TO :' + saveFile)
        
        out = cv2.VideoWriter(saveFile, cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))

        if not cap.isOpened():
            print("Error opening video stream or file")
        
        tracker = sort.Sort(max_age = self.trackMaxAge, min_hits= self.trackMinHits, iou_threshold = self.trackIouThreshold)
        
        totalCount = []
        # currentArray = np.empty((0, 5))
        
        while cap.isOpened():
            detections = np.empty((0, 5))
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading frame")
                break
            
            beg = time.time()
            results = self.predictImg(frame, verbose = False)
            if results == None:
                print('***********************************************')
            fps = 1 / (time.time() - beg)
            for box in results.boxes:
                score = box.conf.item() * 100
                classId = int(box.cls.item())

                # x1 , y1 , x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)
                x1, y1, x2, y2 = np.squeeze(box.xyxy.cpu().numpy()).astype(int)

                # currentArray = np.array([x1, y1, x2, y2, score])
                # detections = np.vstack((detections, currentArray))
                detections = np.vstack((detections, [x1, y1, x2, y2, score]))

            resultsTracker = tracker.update(detections)
            
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                id_txt = f"ID: {str(id)}"
                cv2.putText(frame, id_txt, (cx, cy), 4, 0.5, (0, 0, 255), 1)

                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    
            if display == 'default':
                frame = self.defaultDisplay(**display_args)
            
            elif display == 'custom':
                frame == self.customDisplay( **display_args)

            frame = cv2.putText(frame,f"FPS : {fps:,.2f}", (5,55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
            
            count_txt = f"TOTAL COUNT : {len(totalCount)}"
            frame = cv2.putText(frame, count_txt, (5,45), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            # cv2.imshow("Counting",frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        print(len(totalCount))
        print(totalCount)