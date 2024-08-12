# This script can only detect gestures of the right hand
import cv2
import time
import os
import HandTrackingModule as htm

# Set the width and height of the camera, i.e., resolution
wCam, hCam = 640, 480

# Open the camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # Set camera width
cap.set(4, hCam)  # Set camera height

# Specify the path to the finger image folder
folderPath = "E:\\Advance Computer Vision with Python\\main_en\\Project 2 Finger Counter\\FingerImages"
myList = os.listdir(folderPath)  # Get a list of all file names in the folder
print(myList)

overlayList = []  # List to store finger images
for imPath in myList:
    # Read each finger image
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)  # Add image to the list

print(len(overlayList))  # Output the number of images

pTime = 0  # Initialize previous frame time

# Create a hand detector object, set detection confidence to 0.75
detector = htm.handDetector(detectionCon=0.75)

# List of fingertip IDs
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()  # Read frame from camera
    img = cv2.flip(img, 1)  # Horizontally flip the image
    img = detector.findHands(img)  # Detect hands and draw hand keypoints
    lmList = detector.findPosition(
        img, draw=False
    )  # Get list of hand keypoint positions

    if len(lmList) != 0:
        fingers = []

        # Detect thumb (based on x-coordinates of thumb tip and second joint)
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            # If following the source code, if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] determines the extension of the right thumb before flipping
            # So if I choose img = cv2.flip(img, 1), then if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] determines the extension of the left thumb after horizontal flipping, to keep detecting the right hand, change greater than to less than
            # Therefore, this script can only detect gestures of the right hand
            fingers.append(1)  # 1 indicates the finger is extended
        else:
            fingers.append(0)  # 0 indicates the finger is bent

        # Detect the other four fingers (based on y-coordinates of fingertip and finger root)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)  # Count the number of extended fingers
        print(totalFingers)

        # Select the image corresponding to the number of fingers and overlay it
        h, w, c = overlayList[totalFingers - 1].shape
        # totalFingers - 1 is because list indices start from 0, while totalFingers represents the number of fingers (starting from 1). So subtract 1 to correctly access the corresponding image in the list
        # Therefore, put 0 at the end of the list, because at this time no fingers are extended, totalFingers is 0, subtracting 1 gets -1, which is the last one in the list
        img[0:h, 0:w] = overlayList[
            totalFingers - 1
        ]  # Place the image of the corresponding number of fingers in the top-left corner of the camera image

        # Draw a rectangle and display the number of fingers
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            img,
            str(totalFingers),
            (45, 375),
            cv2.FONT_HERSHEY_PLAIN,
            10,
            (255, 0, 0),
            25,
        )

    cTime = time.time()  # Get current frame time
    fps = 1 / (cTime - pTime)  # Calculate frame rate
    pTime = cTime  # Update previous frame time

    # Display frame rate on the image
    cv2.putText(
        img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    # Display the processed image
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # Wait for a key press, 1 millisecond
