import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame reduction
smoothening = 7  # Smoothing factor
#########################
# The above are comments used to create visual separators in the code. They can help developers to more clearly separate different parts of the code, making it more readable. These lines themselves have no function, just to improve code readability

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()  # Get screen width and height
# autopy.screen.size() returns the screen size in pixels
# wScr is the screen width, hScr is the screen height
# print(wScr, hScr)

while True:
    # 1. Detect hand landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Horizontally flip the image
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get positions of index and middle fingertips
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(
            img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2
        )

        # 4. Only index finger up: Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # np.interp() is a function for linear interpolation. Its basic usage: np.interp(x, xp, fp)
            # Where x is the point to interpolate, xp is the range of input data points (known x values), fp is the range of output data points (known y values)
            # Here, x1 and y1 are the positions of the finger in the camera image, (frameR, wCam - frameR) is the range of finger movement in the camera image, (0, wScr) and (0, hScr) are the screen coordinate ranges
            # Through np.interp(), the program maps the finger position in the camera image to the screen coordinates, thus achieving the correspondence between finger movement and mouse cursor movement. This allows finger movement in the camera to control the mouse cursor on the screen

            # 6. Smooth the values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # clocX and clocY are the current smoothed coordinates, plocX and plocY are the previous frame coordinates
            # x3 and y3 are the target coordinates after interpolation in the current frame, smoothening is a smoothing factor to control the degree of smoothing
            # Each update, the current position (clocX, clocY) moves a small part towards the target position (x3, y3), smoothening controls the step size of the movement, the larger the value, the slower and smoother the movement
            # Smoothing makes the mouse pointer move a bit slower, but the purpose is to reduce jitter and make the movement smoother
            # If you feel the movement is too slow, you can try reducing the value of smoothening, so the pointer will follow the finger movement faster

            # 7. Move the mouse
            # autopy.mouse.move(wScr - clocX, clocY)
            # Use the above statement if not flipped
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both index and middle fingers up: Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Calculate the distance between the two fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)

            # 10. If the distance is short, click the mouse
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
