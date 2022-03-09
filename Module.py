
import cv2
from matplotlib.pyplot import draw
import mediapipe as mp
import time
import os
import numpy as np
class pose_detector():
    def __init__(self,mode=False,smooth=True,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon= detectionCon 
        self.trackCon= trackCon
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,smooth_landmarks=self.smooth,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)      
    
    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    def findlmList(self,img):
        self.lmList=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h , w , c =img.shape
                cx , cy= int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
        return self.lmList
    def find_particular_angle(self,img,p1,p2,p3,draw=True):
        self.angle=None
        if self.lmList:
            x1,y1 = self.lmList[p1][1:]
            x2,y2 = self.lmList[p2][1:]
            x3,y3 = self.lmList[p3][1:]
            a = [x1,y1]
            b = [x2,y2]
            c = [x3,y3]
            a = np.array(a) # Shoulder
            b = np.array(b) # Elbow
            c = np.array(c) # Wrist
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            self.angle =  np.abs(radians*180.0/np.pi)
            if self.angle >180:
                self.angle =360-self.angle
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
                cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
                cv2.putText(img, str(int(self.angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return self.angle

if __name__ == "__main__":
    counter=0
    test_video = os.path.join("test","video.mp4")
    cap = cv2.VideoCapture(test_video)
    pTime=0
    detect = pose_detector()
    while cap.isOpened():
        ret,frame = cap.read()
        frame= cv2.resize(frame,(500,500))
        frame = detect.findPose(frame,draw=False)
        lmlist= detect.findlmList(frame)

        if len(lmlist)>0:

            #angle = detect.find_particular_angle(frame,12,14,16,draw=True)

            angle = detect.find_particular_angle(frame,11,13,15,draw=True)
            if angle > 150:
                stage = "down"
            if angle < 100 and stage =='down':
                stage="up"
                counter +=1
                print(counter)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, "counter : "+str(int(counter)), (70, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
        cv2.putText(frame, "fps : "+str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()