import cv2
import mediapipe as mp
import time

# Open video file
cap = cv2.VideoCapture(
    "E:\\Advance Computer Vision with Python\\main\\Chapter 3 Face Detection\\Videos\\4.mp4"
)

pTime = 0  # Previous frame time

# Initialize MediaPipe face detection module
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(
    0.75
)  # Create a MediaPipe face detector object with a detection confidence threshold of 0.75

while True:
    success, img = cap.read()  # Read video frame
    if not success:
        print("Failed to read frame")
        break

    # Convert image to RGB format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)  # Process the image for face detection

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection) # Use MediaPipe tools to draw bounding boxes and keypoints on the detected face
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

            # Get bounding box information for face detection
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # Draw bounding box
            cv2.rectangle(img, bbox, (255, 0, 255), 2)

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
            # The first 5 is font size, the second 5 is font thickness

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
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
