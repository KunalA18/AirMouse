import cv2
import numpy as np
import time
import HandTracking as ht
import mouse

# setting camera properties
wcam, hcam = 540, 440
video = cv2.VideoCapture(0)
video.set(3, wcam)
video.set(4, hcam)

ROI = 100
diluting = 8
pTime = 0
scrnW, scrnH = 1920, 1080

# Mouse locations
prev_X, prev_Y = 0, 0
curr_X, curr_Y = 0, 0

detector = ht.HandTracker(maxHands=1) 

while(video.isOpened()):
    # Finding hand landmarks
    success, frame = video.read()
    frame = detector.detectHands(frame)
    lmlist, boundary = detector.findPosition(frame)

    # Getting the tip of middle and index finger
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

    # Checking which finger is up
        fingers = detector.check_fingers()
        cv2.rectangle(frame, (ROI+20, ROI-20),
                      (wcam-ROI, hcam-ROI), (0, 0, 255), 2)
    # Moving when index finger is up
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert co-ordinates
            x3 = np.interp(x1, (ROI-20, wcam-100), (0, scrnW))
            y3 = np.interp(y1, (ROI, hcam-60), (0, scrnH))

        # diluting values
            curr_X = prev_X + (x3-prev_X)/diluting
            curr_Y = prev_Y + (y3-prev_Y)/diluting

        # moving mouse
            mouse.move(scrnW-curr_X, curr_Y)
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            prev_X, prev_Y = curr_X, curr_Y

        # CLick when both index and middle fingers are up and distance between them is less
        if fingers[1] == 1 and fingers[2] == 1:
            # Finding distance between fingers
            length, frame, line = detector.findDistance(8, 12, frame)
            if length < 25:
                cv2.circle(frame, (line[4], line[5]),
                           10, (0, 255, 0), cv2.FILLED)
                mouse.click()
                time.sleep(0.2)

    # Frame 
    cTime = time.time()
    fps = 1/(cTime-pTime) 
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Airmouse_kunal", frame)
    if cv2.waitKey(10) == ord('x'): 
        cv2.destroyAllWindows()
        video.release()
        break
