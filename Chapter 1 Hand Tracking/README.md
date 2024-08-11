# mp.solutions.hands

`mpHands = mp.solutions.hands` refers to the MediaPipe Hands solution module.

`hands = mpHands.Hands()` creates a Hands object for detecting and tracking hand landmarks.

`results = hands.process(imgRGB)` processes the image to detect hands.

`for handLms in results.multi_hand_landmarks` returns a list containing landmark information for each detected hand.

`for id, lm in enumerate(handLms.landmark)` where `id` is the index of the hand landmark, and `lm` is short for landmark, representing the coordinates of the hand landmark.

`lm.x` and `lm.y` are the normalized coordinates of the landmarks, ranging from 0 to 1. By multiplying them by the image's width and height, they can be converted to pixel coordinates within the image.

Parameters for the `mpHands.Hands()` object:

```python
static_image_mode: Whether to treat each frame as a static image
max_num_hands: Maximum number of hands to detect
min_detection_confidence: Detection confidence threshold
min_tracking_confidence: Tracking confidence threshold
```

# mp.solutions.drawing_utils

`mpDraw = mp.solutions.drawing_utils` refers to the drawing utilities for drawing detected hand landmarks and connections on the image.

`mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)` draws hand landmarks and connections. `img` is the image to draw on, `handLms` are the hand landmark coordinates, and `mpHands.HAND_CONNECTIONS` defines the connections between hand landmarks for drawing the skeleton structure.

# pycache Folder

When a .py file is imported, Python compiles it and stores the compiled bytecode in the `__pycache__` directory. This improves execution efficiency, as the compiled bytecode can be used directly in subsequent runs without recompiling the source code.

"Cache" refers to a temporary storage mechanism for data to allow faster access. Caching can reduce the need for repeated calculations or reading from slower storage devices, thus improving program performance.

For `HandTrackingModule.py`, when you import this module in other files, only the `handDetector` class and its methods are used, while the `main()` function will not be executed. The `main()` function only runs when the script is executed directly.

# Notes

If you encounter the following error:

> cv2.error: OpenCV(4.10.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1301: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x, or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'

Please run the terminal from the outermost folder (for VSCode, open the outermost folder). I'm not entirely sure why this bug occurs.