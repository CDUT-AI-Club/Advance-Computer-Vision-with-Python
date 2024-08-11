import cv2
import mediapipe as mp
import time
import math


# Pose detection class
class poseDetector:
    # The following is old code; some parameters are deprecated
    # def __init__(
    #     self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5
    # ):
    #     # Initialize parameters
    #     self.mode = mode  # Static image mode
    #     self.upBody = upBody  # Whether to detect only the upper body
    #     self.smooth = smooth  # Smoothing
    #     self.detectionCon = detectionCon  # Detection confidence
    #     self.trackCon = trackCon  # Tracking confidence
    #     self.mpDraw = mp.solutions.drawing_utils  # Mediapipe drawing tools
    #     self.mpPose = mp.solutions.pose  # Mediapipe pose detection module
    #     self.pose = self.mpPose.Pose(
    #         self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon
    #     )

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        # Initialize parameters for the pose detector
        self.static_image_mode = static_image_mode  # Whether to process static images; False for video stream
        self.model_complexity = model_complexity  # Model complexity: 0, 1, or 2; higher is more accurate but slower
        self.enable_segmentation = enable_segmentation  # Whether to enable segmentation
        self.min_detection_confidence = (
            min_detection_confidence  # Minimum confidence for pose detection
        )
        self.min_tracking_confidence = (
            min_tracking_confidence  # Minimum confidence for pose tracking
        )

        # Set up MediaPipe tools
        self.mpDraw = mp.solutions.drawing_utils  # MediaPipe drawing tools
        self.mpPose = mp.solutions.pose  # MediaPipe pose estimation module

        # Create pose estimation object
        self.pose = self.mpPose.Pose(
            static_image_mode=self.static_image_mode,  # Static image mode setting
            model_complexity=self.model_complexity,  # Model complexity setting
            enable_segmentation=self.enable_segmentation,  # Segmentation setting
            min_detection_confidence=self.min_detection_confidence,  # Detection confidence threshold
            min_tracking_confidence=self.min_tracking_confidence,  # Tracking confidence threshold
        )

    def findPose(self, img, draw=True):
        # Convert image from BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process image to detect pose
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                # Draw pose connections
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )
        return img

    def findPosition(self, img, draw=True):
        # Initialize list to store landmark positions
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(
                    lm.y * h
                )  # Calculate landmark position in the image
                self.lmList.append([id, cx, cy])
                if draw:
                    # Draw landmarks on the image
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get landmark coordinates
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate angle
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        # Two vectors (x1-x2, y1-y2), (x3-x2, y3-y2)
        # Use atan2 to calculate the angle of each vector with the x-axis
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(
                img,
                str(int(angle)),
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
        return angle


# Main function
def main():
    cap = cv2.VideoCapture(
        "E:\\Advance Computer Vision with Python\\main\\Chapter 2 Pose Estimation\\PoseVideos\\5.mp4"
    )  # Open video file
    pTime = 0  # Previous frame time
    detector = poseDetector()  # Create pose detector
    while True:
        success, img = cap.read()  # Read video frame
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        img = detector.findPose(img)  # Detect pose
        if not success:
            print("Failed to read frame")
            break
        lmList = detector.findPosition(img, draw=False)  # Get landmark positions
        if len(lmList) != 0:
            print(lmList[14])  # Print landmark information
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Calculate frame rate
        pTime = cTime
        cv2.putText(
            img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
        )
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.imshow("Image", img)  # Display image
        cv2.waitKey(1)  # Wait for key to display the next frame


if __name__ == "__main__":
    main()
