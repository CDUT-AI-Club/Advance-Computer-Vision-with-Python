import cv2
import mediapipe as mp
import time

# Open the video file
cap = cv2.VideoCapture(
    "E:\\Advance Computer Vision with Python\\main\\Chapter 3 Face Detection\\Videos\\4.mp4"
)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

pTime = 0

# Initialize MediaPipe drawing tools and face mesh model
mpDraw = mp.solutions.drawing_utils  # Import MediaPipe drawing utilities
mpFaceMesh = (
    mp.solutions.face_mesh
)  # Import MediaPipe face mesh module for detecting and processing facial landmarks
faceMesh = mpFaceMesh.FaceMesh(
    max_num_faces=2
)  # Initialize face mesh model, set to detect up to two faces
drawSpec = mpDraw.DrawingSpec(
    thickness=1, circle_radius=2
)  # Create a drawing specification object for landmark and connection styles
# thickness specifies line thickness, circle_radius specifies landmark point radius

while True:
    print("Reading video frame...")
    success, img = cap.read()
    print("Read success:", success)
    if not success:
        print("Finished processing video or error occurred.")
        break

    # Convert the image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the image to detect face mesh
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # Draw the face mesh
            mpDraw.draw_landmarks(
                img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec
            )
            # mpDraw.draw_landmarks calls MediaPipe's drawing function to draw landmarks and connections on the image
            # img: the image on which to draw the landmarks
            # faceLms: the detected face landmarks
            # mpFaceMesh.FACEMESH_TESSELATION: specifies the type of connections to draw, here it's face mesh tessellation
            # drawSpec: defines the drawing style for landmarks and connections (e.g., thickness and circle radius)
            # The last two drawSpec parameters define the drawing style for landmarks (keypoints) and connections
            # You can choose to define them separately:
            # drawSpecPoints = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)  # Green
            # drawSpecLines = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1)  # Blue
            # Note that circle_radius in drawSpecLines has no effect; it only affects the drawing of landmarks, not connections

            for id, lm in enumerate(faceLms.landmark):
                # Get the image dimensions
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                # Print each landmark's ID and coordinates
                print(id, x, y)

    # Calculate and display the frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create a resizable window

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
