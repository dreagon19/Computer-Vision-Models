import HandTrackingModule as htm
import cv2
import mediapipe as mp
import os

# adding the images to a alist
path = "FingerImg"  
imgList = os.listdir(path)
print(imgList)

#adding their path in overlayList   
overlayList = []
for i in imgList:
    image = cv2.imread(f'{path}/{i}')
    overlayList.append(image)


cap = cv2.VideoCapture(0)
cap.set(3,1200)
cap.set(4,720)


#detection the fingers 

detector = htm.handDetector(detectionCon=0.7)
tips = [4,8,12,16,20]


#fingers Name
name = ["thumb","index","middle","ring","pinky"]
pos = ["0","400","450","500","550"]

while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=True)
    img = cv2.flip(img,1)

    #thumb lmlist[][1] here 1 is x axis

    if len(lmList)!=0:
        fingers =[]
        if lmList[tips[0]][1]>lmList[tips[0]-1][1]:
            cv2.putText(img,name[0],(45,350),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
            fingers.append(1)
        else:
           fingers.append(0)

        #iterate over the list to find total number of finger up
        for id in range(1,5):
            if lmList[tips[id]][2]<lmList[tips[id]-2][2]:#here 2 represent Y axis
                cv2.putText(img,name[id],(45,int(pos[id])),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
                fingers.append(1)
            else:
                fingers.append(0)    

        totalFingers = fingers.count(1)
        #print(totalFingers)


    
        #putting shapes on the screen
        h,w,c = overlayList[totalFingers].shape
        img[0:h,0:w] = overlayList[totalFingers]
    
    
    cv2.imshow("Cam",img)
    if cv2.waitKey(1) == ord('q'):
        break


