import mediapipe as mp
import time
import cv2


class handDetector():
    def __init__(self,mode = False,maxHands = 2,modelComplexity=1,detectionCon = 0.5,trackCon = 0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCoplex = modelComplexity
        self.trackCon = trackCon
        self.detectionCon = detectionCon
        self.mp_hands= mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,self.maxHands,self.modelCoplex,self.detectionCon,self.trackCon)
        self.mp_drawing = mp.solutions.drawing_utils


    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        #imgBGR = cv2.cvtColor(imgRGB,cv2.COLOR_RGB2BGR)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img,handLms,self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNo=0,draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)

        return lmList



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        # if len(lmList) !=0:
        #     print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        img = cv2.flip(img,1)

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

        cv2.imshow("cam",img)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()                    
                








