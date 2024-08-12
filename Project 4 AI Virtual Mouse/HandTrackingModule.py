"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone

Modified by: Diraw
Date: 20240812
Description:
1. Modified the initialization of the `Hands` object to use named parameters for better clarity and compatibility with the latest version of the mediapipe library. This change ensures that the parameters are correctly mapped to the expected arguments in the `Hands` class.
2. Added a line to flip the image horizontally using `cv2.flip(img, 1)` to ensure the hand movements appear mirrored, which is more intuitive for user interaction
3. Updated the findPosition method to check if xList and yList are not empty before calculating the minimum and maximum values. This prevents errors when no hand landmarks are detected.
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialize hand detector
        self.mode = mode  # Mode: static or dynamic
        self.maxHands = maxHands  # Maximum number of hands to detect
        self.detectionCon = detectionCon  # Detection confidence
        self.trackCon = trackCon  # Tracking confidence

        self.mpHands = mp.solutions.hands  # Mediapipe hand solutions
        # self.hands = self.mpHands.Hands(
        #     self.mode, self.maxHands, self.detectionCon, self.trackCon
        # )
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils  # Drawing utilities
        self.tipIds = [4, 8, 12, 16, 20]  # Tip IDs

    def findHands(self, img, draw=True):
        # Find hands and draw
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        self.results = self.hands.process(imgRGB)  # Process image

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )  # Draw hand connections

        return img

    def findPosition(self, img, handNo=0, draw=True):
        # Find hand position
        xList = []  # X-coordinate list
        yList = []  # Y-coordinate list
        bbox = []  # Bounding box
        self.lmList = []  # Store hand landmarks

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # Get image dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert to pixel coordinates
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(
                        img, (cx, cy), 5, (255, 0, 255), cv2.FILLED
                    )  # Draw landmarks

        if xList and yList:  # Check if lists are not empty
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax  # Calculate bounding box

            if draw:
                cv2.rectangle(
                    img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2
                )  # Draw bounding box

        return self.lmList, bbox

    def fingersUp(self):
        # Determine if fingers are up
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            # As in previous projects, for right hand before flip use if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            # After flip, change greater than to less than
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        # Calculate distance between two points
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoint coordinates

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)  # Draw line
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)  # Draw start point
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)  # Draw end point
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)  # Draw midpoint
            length = math.hypot(x2 - x1, y2 - y1)  # Calculate distance

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # Main function
    pTime = 0  # Previous frame time
    cTime = 0  # Current time
    cap = cv2.VideoCapture(0)  # Open camera
    detector = handDetector()  # Create hand detector

    while True:
        success, img = cap.read()  # Read camera image
        img = cv2.flip(img, 1)  # Horizontally flip image
        img = detector.findHands(img)  # Find hands
        lmList, bbox = detector.findPosition(img)  # Get hand position
        if len(lmList) != 0:
            print(lmList[4])  # Print thumb position

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Calculate frame rate
        pTime = cTime

        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )  # Display frame rate

        cv2.imshow("Image", img)  # Display image
        cv2.waitKey(1)  # Wait for key press


if __name__ == "__main__":
    main()  # Run main function
