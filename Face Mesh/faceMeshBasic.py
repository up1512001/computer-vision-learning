import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpac = mpDraw.DrawingSpec(thickness=1,circle_radius=1)

while True:

    success , img = cap.read()

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACE_CONNECTIONS,drawSpac,drawSpac)

            for id,lm in enumerate(faceLms.landmark):
                # print(lm)
                ih,iw,ic = img.shape
                cx,cy = int(lm.x*iw),int(lm.y*ih)
                print(id,cx,cy)





    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f'FPS:{int(fps)}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    cv2.imshow("Image :",img)
    cv2.waitKey(1)

