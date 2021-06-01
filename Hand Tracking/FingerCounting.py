import cv2
import os
import time

import HandTrackingModule as htm

wCam,hCam = 640 , 480

cap = cv2.VideoCapture(0)
cap.set(1,wCam)
cap.set(2,hCam)

folderpath = "C:/Users/om/PycharmProjects/HandTracking/Images"
myList = os.listdir(folderpath)
# print(myList)
overlayList = []
for impath in myList:
    image = cv2.imread(f'{folderpath}/{impath}')
    # print(f'{folderpath}/{impath}')
    overlayList.append(image)

# print(len(overlayList))

detector = htm.HandDectector(detectionCon=0.75)

pTime = 0

tipIds = [4,8,12,16,20]

while True:

    success,img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []
        # Thumb left hand
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            # print("Index Finger Open...")
            fingers.append(1)
        else:
            fingers.append(0)

        # four fingers
        for id in range(1,5):

            # maximum hight is 0 and lowest is bottem
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                # print("Index Finger Open...")
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        h,w,c = overlayList[totalFingers-1].shape
        img[0:h,0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,8,(255,0,0),25)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img,f'FPS:{int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image :",img)
    cv2.waitKey(1)
