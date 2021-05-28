import cv2
import mediapipe
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
dectector = htm.HandDectector()
while True:
    success, img = cap.read()
    img = dectector.findHands(img,draw=False)

    lmList = dectector.findPosition(img,draw=False);
    if len(lmList) != 0:
        print(lmList[0],lmList[4],lmList[8],lmList[12],lmList[16])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 25), 3)

    cv2.imshow("Image :", img)
    cv2.waitKey(1)
