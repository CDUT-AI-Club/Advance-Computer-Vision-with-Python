import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# from pprint import pprint

# Set camera width and height
wCam, hCam = 640, 480

# Initialize camera
cap = cv2.VideoCapture(0)
# if cap.isOpened():
#     print("Camera successfully opened")
# else:
#     print("Failed to open camera")
cap.set(3, wCam)  # Set width
cap.set(4, hCam)  # Set height
# Set the resolution of the camera capture to 640x480, i.e., 640x480 pixels

pTime = 0  # Time of the previous frame

# Initialize hand detector with detection confidence
detector = htm.handDetector(detectionCon=0.7)

# Get audio devices
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]  # Minimum volume
maxVol = volRange[1]  # Maximum volume
# print(volRange)  # (-45.0, 0.0, 1.0)

vol = 0  # Current volume
volBar = 400  # Volume bar position
volPer = 0  # Volume percentage

while True:
    success, img = cap.read()  # Read camera image
    # if success:
    #     print("Image captured successfully")
    #     # cv2.imshow("Captured Image", img)
    #     # cv2.waitKey(0)
    # else:
    #     print("Failed to capture image")
    img = cv2.flip(img, 1)  # Flip image horizontally
    img = detector.findHands(img)  # Detect hands
    lmList = detector.findPosition(img, draw=False)  # Get list of hand landmarks
    # print(type(lmList))
    # pprint(lmList)
    # lmList_is_empty = all(len(lst) == 0 for lst in lmList)
    lmList = lmList[0]
    # pprint(lmList)

    if len(lmList) != 0:
        # Get coordinates of index finger and thumb
        try:
            if len(lmList) > 4 and len(lmList[4]) > 1:
                x1, y1 = lmList[4][1], lmList[4][2]
            else:
                raise IndexError("Landmark index out of range")
        except IndexError as e:
            print(e)
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point
        # print(1)

        # Draw circles and line
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # Thumb circle
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)  # Index finger circle
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Connecting line
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # Center point circle

        # Calculate distance between two points
        length = math.hypot(x2 - x1, y2 - y1)

        # Convert distance to volume
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        print(int(length), vol)

        # Set volume
        volume.SetMasterVolumeLevel(vol, None)

        # If distance is less than 50, change circle color
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)  # Outer frame
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)  # Fill
    cv2.putText(
        img, f"{int(volPer)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3
    )

    # Calculate frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display frame rate
    cv2.putText(
        img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3
    )

    # Show image
    cv2.imshow("Img", img)
    cv2.waitKey(1)
