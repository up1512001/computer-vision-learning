import cv2
import mediapipe as mp
import time
import PoseModule as pm
import numpy as np

cap = cv2.VideoCapture(0)
pTime = 0

detector = pm.poseDetector()

count = 0
dir = 0

while True:

    success, img = cap.read()

    img = detector.findPose(img,False)
    lmList = detector.getPosition(img,False)

    if len(lmList) != 0:
        # print(lmList)
        # right arm
        # detector.findAngle(img,12,14,16)

        # left arm
        angle = detector.findAngle(img,11,13,15)
        per = np.interp(angle,(210,310),(0,100))

        bar = np.interp(angle,(210,310),(650,100))

        # print("Presentage :->",per)

        # check for dumbell curl
        color = (255,0,255)
        if per == 100:
            color = (0,255,0)
            if dir==0:
                count+=0.5
                dir = 1

        if per == 0:
            color = (0,0,255)
            if dir == 1:
                count+=0.5
                dir=0

        # print("Count :->",count)
        print(img.shape)
        # bar drawing
        cv2.rectangle(img,(470,100),(600,550),color,2)
        cv2.rectangle(img, (470, int(bar)), (600, 550), color, cv2.FILLED)
        cv2.putText(img,f'{int(per)}%',(470,75),cv2.FONT_HERSHEY_PLAIN,4,color,4)

        # draw count
        cv2.rectangle(img,(0,450),(150,290),(0,255,0),cv2.FILLED)

        cv2.putText(img,str(int(count)),(5,452),cv2.FONT_HERSHEY_PLAIN,15,(255,0,0),24)

        # # right foot
        # detector.findAngle(img,24,26,28)
        #
        # # left foot
        # detector.findAngle(img,23,25,27)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img,f'FPS:{int(fps)}',(10,50),cv2.FONT_ITALIC,2,(0,0,255),2)

    cv2.imshow("Image :",img)
    cv2.waitKey(1)
