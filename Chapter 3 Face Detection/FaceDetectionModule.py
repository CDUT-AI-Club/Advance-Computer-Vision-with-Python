import cv2
import mediapipe as mp
import time


# Define a face detection class
class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        # Initialize detection confidence
        self.minDetectionCon = minDetectionCon

        # Initialize MediaPipe face detection module
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        # Convert image to RGB format
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)  # Perform face detection
        bboxs = []  # Store detected bounding boxes

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # Get bounding box information for face detection
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                bboxs.append([id, bbox, detection.score])  # Add to bounding box list

                if draw:
                    img = self.fancyDraw(img, bbox)  # Draw bounding box
                    # Display detection confidence
                    cv2.putText(
                        img,
                        f"{int(detection.score[0] * 100)}%",
                        (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        5,
                        (255, 0, 255),
                        5,
                    )
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        # The fancyDraw function adds custom styling by drawing short lines at the corners of the rectangle,
        # adding visual variation compared to a regular rectangle.
        # 1. Rectangle: Use cv2.rectangle to draw a standard rectangle.
        # 2. Corner lines: Draw short line segments at the four corners of the rectangle for a designed look.
        # l is the length of corner lines, t is the thickness of corner lines, rt is the thickness of the rectangle (length, thickness).

        # Custom drawing style for the border
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)  # Draw rectangle
        # Draw lines at the four corners
        # Top left corner
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top right corner
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom left corner
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom right corner
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    # Open video file
    cap = cv2.VideoCapture(
        "E:\\Advance Computer Vision with Python\\main\\Chapter 3 Face Detection\\Videos\\4.mp4"
    )
    pTime = 0  # Previous frame time
    detector = FaceDetector()  # Create face detector object

    while True:
        success, img = cap.read()  # Read video frame
        img, bboxs = detector.findFaces(img)  # Detect faces and get bounding boxes
        print(bboxs)  # Print bounding box information

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5
        )

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create a resizable window

        # Display image
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
