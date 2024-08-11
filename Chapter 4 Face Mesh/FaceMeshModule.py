import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(
        self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5
    ):
        # Initialize parameters
        self.staticMode = staticMode  # Use static mode or not
        self.maxFaces = maxFaces  # Maximum number of faces to detect
        self.minDetectionCon = minDetectionCon  # Minimum detection confidence
        self.minTrackCon = minTrackCon  # Minimum tracking confidence

        # Initialize MediaPipe drawing tools and face mesh model
        self.mpDraw = mp.solutions.drawing_utils  # Drawing utilities
        self.mpFaceMesh = mp.solutions.face_mesh  # Face mesh module
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon,
        )

        self.drawSpec = self.mpDraw.DrawingSpec(
            thickness=1, circle_radius=2
        )  # Drawing specifications

    def findFaceMesh(self, img, draw=True):
        # Convert the image to RGB
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image to detect face mesh
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []  # Store detected facial landmarks
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    # Draw the face mesh
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.drawSpec,
                        self.drawSpec,
                    )
                face = []  # Store landmarks for a single face
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape  # Get image dimensions
                    x, y = int(lm.x * iw), int(
                        lm.y * ih
                    )  # Convert normalized coordinates to pixel coordinates
                    face.append([x, y])  # Add landmark coordinates
                faces.append(face)  # Add to faces list
        return img, faces  # Return image and facial landmarks


def main():
    # Open the video file
    cap = cv2.VideoCapture(
        "E:\\Advance Computer Vision with Python\\main\\Chapter 3 Face Detection\\Videos\\4.mp4"
    )
    pTime = 0  # Previous frame time
    detector = FaceMeshDetector(maxFaces=2)  # Initialize face mesh detector
    while True:
        success, img = cap.read()
        if not success:
            break
        img, faces = detector.findFaceMesh(img)  # Detect face mesh
        if len(faces) != 0:
            print(faces[0])  # Print landmarks of the first face
        cTime = time.time()  # Current time
        fps = 1 / (cTime - pTime)  # Calculate frames per second
        pTime = cTime  # Update previous frame time
        cv2.putText(
            img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
        )  # Display FPS on the image
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.imshow("Image", img)  # Show image
        cv2.waitKey(1)  # Wait for keyboard input


if __name__ == "__main__":
    main()
