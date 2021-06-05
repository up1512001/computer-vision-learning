import cv2
import mediapipe as mp
import time
import os
import HandTrackingModule as htm
import numpy as np

folderPath = 'Images for virtual painter'
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))
###########################################
brushThikness = 10
eraserThikness = 100
xp,yp = 0,0
###########################################
header = overlayList[0]
drawColor = (255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.HandDectector(detectionCon=0.75)

imCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    # 1. import image
    success , img = cap.read()
    img = cv2.flip(img,1)

    #  2. find hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        # 3. check which fingers are up

        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If selection mode -two fingers are up
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            print('Selection Mode')

            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550 < x1 < 750 :
                    drawColor = (255,0,0)
                    header = overlayList[1]
                elif 800 < x1 < 950:
                    drawColor = (0,255,0)
                    header = overlayList[2]
                elif 1000 < x1 < 1200:
                    drawColor = (0, 0, 0)
                    header = overlayList[3]

            cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), drawColor, cv2.FILLED)

        # 5. If drawing mode - index finger up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),10,drawColor,cv2.FILLED)
            print('Drawing Mode')

            if xp==0 and yp == 0:
                xp,yp = x1,y1
            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThikness)
                cv2.line(imCanvas, (xp, yp), (x1, y1), drawColor, eraserThikness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThikness)
                cv2.line(imCanvas, (xp, yp), (x1, y1), drawColor, brushThikness)

            xp,yp = x1,y1


    imgGray = cv2.cvtColor(imCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)

    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imCanvas)

    # print(img.shape)
    # setting header image
    img[0:125,0:1280] = header
    # img = cv2.addWeighted(img,0.5,imCanvas,0.5,0)
    cv2.imshow('Image :',img)
    # cv2.imshow('Image Canvas :',imCanvas)
    # cv2.imshow('Image Inverse :',imgInv)
    cv2.waitKey(1)
