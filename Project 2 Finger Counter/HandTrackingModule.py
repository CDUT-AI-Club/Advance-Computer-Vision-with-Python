"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone

Modified by: Diraw
Date: 20240812
Description:
1. Modified the initialization of the `Hands` object to use named parameters for better clarity and compatibility with the latest version of the mediapipe library. This change ensures that the parameters are correctly mapped to the expected arguments in the `Hands` class.
2. Added a line to flip the image horizontally using `cv2.flip(img, 1)` to ensure the hand movements appear mirrored, which is more intuitive for user interaction
"""

import cv2
import mediapipe as mp
import time


# Hand detector class
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialize parameters
        self.mode = mode  # Whether to use static mode
        self.maxHands = maxHands  # Maximum number of hands to detect
        self.detectionCon = detectionCon  # Detection confidence
        self.trackCon = trackCon  # Tracking confidence

        # Initialize MediaPipe hand model
        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(
        #     self.mode, self.maxHands, self.detectionCon, self.trackCon
        # )
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )

        # Initialize drawing tools
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert the image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image to detect hands
        self.results = self.hands.process(imgRGB)

        # If hands are detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # Draw hand keypoints and connections
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []  # Store hand keypoint positions
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # Get image dimensions
                h, w, c = img.shape
                # Calculate pixel position of keypoints
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                # Draw keypoints
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList


# Main function
def main():
    pTime = 0  # Previous frame time
    cTime = 0  # Current frame time
    cap = cv2.VideoCapture(0)  # Open the camera
    detector = handDetector()  # Create hand detector object

    while True:
        success, img = cap.read()  # Read camera frame
        img = cv2.flip(img, 1)  # Horizontally flip the image
        img = detector.findHands(img)  # Detect hands and draw
        lmList = detector.findPosition(img)  # Get hand keypoint positions
        if len(lmList) != 0:
            print(lmList[4])  # Print coordinates of the thumb tip

        cTime = time.time()  # Get current time
        fps = 1 / (cTime - pTime)  # Calculate frame rate
        pTime = cTime  # Update previous frame time

        # Display frame rate on the image
        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        # Display the image
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
