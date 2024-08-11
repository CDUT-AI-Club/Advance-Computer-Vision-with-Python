import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0  # Previous frame time
cTime = 0  # Current time
cap = cv2.VideoCapture(0)  # Open the camera
detector = htm.handDetector()  # Create a hand detector

while True:
    success, img = cap.read()  # Read camera image
    img = detector.findHands(img, draw=True)  # Detect hands and draw connections
    lmList = detector.findPosition(img, draw=False)  # Get hand landmark positions

    if len(lmList) != 0:
        print(lmList[4])  # Print thumb tip position

    cTime = time.time()
    fps = 1 / (cTime - pTime)  # Calculate frame rate
    pTime = cTime

    # Display frame rate on the image
    # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    #             (255, 0, 255), 3)

    cv2.imshow("Image", img)  # Show image
    cv2.waitKey(1)  # Wait for 1 ms
