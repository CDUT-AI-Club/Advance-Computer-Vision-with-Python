"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone

Modified by: Diraw
Date: 20240811
Description:
1. Modified the initialization of the `Hands` object to use named parameters for better clarity and compatibility with the latest version of the mediapipe library. This change ensures that the parameters are correctly mapped to the expected arguments in the `Hands` class.
2. Added a line to flip the image horizontally using `cv2.flip(img, 1)` to ensure the hand movements appear mirrored, which is more intuitive for user interaction.
3. Added the code lmList = lmList[0] at line 59 in VolumeHandControl.py to fix the error: IndexError: tuple index out of range.
"""

import cv2
import mediapipe as mp
import time
import math


# Hand detection class
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialize parameters
        self.mode = mode  # Static mode flag
        self.maxHands = maxHands  # Maximum number of hands to detect
        self.detectionCon = detectionCon  # Detection confidence threshold
        self.trackCon = trackCon  # Tracking confidence threshold

        # Initialize hand detection module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )

        self.mpDraw = mp.solutions.drawing_utils  # Drawing utilities
        self.tipIds = [4, 8, 12, 16, 20]  # Fingertip IDs

    # Find hands and draw
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        self.results = self.hands.process(imgRGB)  # Process the image

        if self.results.multi_hand_landmarks:  # If hands are detected
            for handLms in self.results.multi_hand_landmarks:
                if draw:  # If drawing is enabled
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    # Find hand position
    def findPosition(self, img, handNo=0, draw=True):
        xList = []  # Store x coordinates
        yList = []  # Store y coordinates
        bbox = []  # Bounding box
        self.lmList = []  # Store hand landmarks

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[
                handNo
            ]  # Select the handNo-th hand
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # Get image dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert to pixel coordinates
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])  # Add to list
                if draw:  # If drawing is enabled
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)  # Calculate bounding box
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:  # Draw bounding box
                cv2.rectangle(
                    img,
                    (bbox[0] - 20, bbox[1] - 20),
                    (bbox[2] + 20, bbox[3] + 20),
                    (0, 255, 0),
                    2,
                )

        return self.lmList, bbox

    # Detect if fingers are up
    def fingersUp(self):
        # In OpenCV, the top-left corner of the image is the origin (0,0), x-coordinate increases to the right, y-coordinate increases downwards
        # If img = cv2.flip(img, 1) is used, the image is flipped horizontally. As a result, the x-coordinate relationship is reversed: the right side of the image becomes the left side of the coordinate system. Therefore, the x-coordinate becomes larger as it moves to the left.
        # Further rational analysis:
        # The statement "In OpenCV, the top-left corner of the image is the origin (0,0), x-coordinate increases to the right, y-coordinate increases downwards" is always true. This means that the image we see on the computer screen always follows this coordinate system.
        # Before flipping, our movements are opposite to what's shown on the computer screen. From our perspective, moving to the right decreases the x-coordinate in the computer image. After flipping the image, from our perspective, moving to the left decreases the x-coordinate in the computer image. Now our perspective aligns with the computer's coordinate system.
        # Based on this, the condition for determining if the right thumb is extended is that the x-coordinate of point 4 is less than the x-coordinate of point 3.

        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            # self.tipIds = [4, 8, 12, 16, 20]  # Fingertip IDs
            # self.tipIds[0] is the index of the thumb tip (4), self.tipIds[0] - 1 is the index of the joint before the thumb tip (3)
            # [1] gets the x-coordinate of that joint from self.lmList, because self.lmList.append([id, cx, cy]), where the 0th dimension is id, and the 1st dimension is the x-coordinate
            # This condition now applies to the case when the right thumb is extended
            fingers.append(1)  # Right thumb is extended, return 1
        else:
            fingers.append(0)  # Right thumb is bent, return 0
        # Other four fingers
        for id in range(1, 5):  # Loop through 4 IDs: 1, 2, 3, 4
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                # This is for judging other fingers. In OpenCV, y-coordinate increases downwards
                fingers.append(1)  # Finger is extended
            else:
                fingers.append(0)  # Finger is bent
        return fingers

    # Calculate distance between two points
    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]  # First point
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]  # Second point
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoint

        if draw:  # If drawing is enabled
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)  # Calculate distance
        return length, img, [x1, y1, x2, y2, cx, cy]


# Main function
def main():
    pTime = 0  # Previous frame time
    cap = cv2.VideoCapture(0)  # Open camera
    detector = handDetector()  # Initialize detector

    while True:
        success, img = cap.read()  # Read image
        img = cv2.flip(img, 1)  # Flip image horizontally
        img = detector.findHands(img)  # Detect hands
        lmList, bbox = detector.findPosition(img)  # Get position
        if len(lmList) != 0:
            print(lmList[4])  # Print specific landmark

        cTime = time.time()  # Current time
        fps = 1 / (cTime - pTime)  # Calculate FPS
        pTime = cTime

        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )  # Display FPS

        cv2.imshow("Image", img)  # Show image
        cv2.waitKey(1)  # Wait for key press


if __name__ == "__main__":
    main()  # Run main function
