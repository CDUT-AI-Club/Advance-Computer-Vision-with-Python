import cv2
import mediapipe as mp
import time

# Initialize Mediapipe's drawing tools and pose detection module
mpDraw = (
    mp.solutions.drawing_utils
)  # Import MediaPipe's drawing tools for drawing detected pose connections and landmarks on the image
mpPose = mp.solutions.pose  # Import MediaPipe's pose estimation module
pose = (
    mpPose.Pose()
)  # Create a pose detection object for processing images and detecting human poses

# Open video file
cap = cv2.VideoCapture(
    "E:\\Advance Computer Vision with Python\\main\\Chapter 2 Pose Estimation\\PoseVideos\\3.mp4"
)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

pTime = 0  # Time of the previous frame

# Create a resizable window
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()  # Read video frame

    if not success:
        print("Failed to read frame")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
    results = pose.process(imgRGB)  # Process image to detect pose

    if results.pose_landmarks:
        # Draw pose connections
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Iterate over each landmark
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape  # Get image dimensions
            cx, cy = int(lm.x * w), int(
                lm.y * h
            )  # Calculate landmark position in the image
            cv2.circle(
                img, (cx, cy), 5, (255, 0, 0), cv2.FILLED
            )  # Draw a circle at the landmark position

    cTime = time.time()
    fps = 1 / (cTime - pTime)  # Calculate frame rate
    pTime = cTime

    # Display frame rate on the image
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
