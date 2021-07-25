import cv2
import numpy as np
import math
import mediapipe as mp
import time


class HandTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.6, trackCon=0.6):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipPos = [4, 8, 12, 16, 20]

    def detectHands(self, frame, draw=True):

        # converting frame to RGB for hands module
        frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(frameRgb)  # detecting hands

        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    # drawing 21 hand-knuckle coordinates and connecting them
                    self.mpDraw.draw_landmarks(
                        frame, handlms, self.mphands.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        self.lmlist = []
        x_coordinates = []
        y_coordinates = []
        boundary = []

        if self.result.multi_hand_landmarks:  # returns true when hand is detected

            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):

                h, w, c = frame.shape
                # finding location of hand in terms of pixel values
                fx, fy = int(lm.x*w), int(lm.y*h)

                x_coordinates.append(fx)
                y_coordinates.append(fy)
                # storing id and pixel co-ordinates of 21 detected points
                self.lmlist.append([id, fx, fy])
                if draw:
                    cv2.circle(frame, (fx, fy), 3, (255, 0, 0), cv2.FILLED)

            xmin, xmax = min(x_coordinates), max(x_coordinates)
            ymin, ymax = min(y_coordinates), max(y_coordinates)
            boundary = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20),
                              (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmlist, boundary

    def check_fingers(self):
        self.fingers = []
        # Thumb
        if self.lmlist[self.tipPos[0]][1] > self.lmlist[self.tipPos[0] - 1][1]:
            self.fingers.append(1)
        else:
            self.fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmlist[self.tipPos[id]][2] < self.lmlist[self.tipPos[id] - 2][2]:
                self.fingers.append(1)
            else:
                self.fingers.append(0)

        return self.fingers

    def findDistance(self, p1, p2, frame, draw=True, r=10, t=3):
        # extracting x,y co-ordinates of tip of index finger
        x1, y1 = self.lmlist[p1][1:]
        # extracting x,y co-ordinates of tip of middle finger
        x2, y2 = self.lmlist[p2][1:]
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            # draw line joining the two finger tips
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), t)
            # draw circle on tip of index finger
            cv2.circle(frame, (x1, y1), r, (255, 0, 0), cv2.FILLED)
            # draw circle on tip of middle finger
            cv2.circle(frame, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            # draw circle on mid of line joining above 2 circles
            cv2.circle(frame, (mx, my), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, mx, my]


def main():
    pTime = 0
    cTime = 0

    video = cv2.VideoCapture(0)
    detector = HandTracker()

    while(video.isOpened()):
        success, frame = video.read()
        frame = detector.detectHands(frame)
        lmlist, boundary = detector.findPosition(frame)


        cTime = time.time()
        fps = 1/(cTime-pTime)  # calculating frame rate
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

        cv2.imshow("Hand", frame)
        if cv2.waitKey(5) == ord('x'):  # press x key to close camera
            cv2.destroyAllWindows()
            video.release()
            break


if __name__ == "__main__":
    main()
