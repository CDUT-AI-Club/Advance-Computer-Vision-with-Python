import cv2
import numpy as np
import os
import HandTrackingModule as htm

# Set the thickness of brush and eraser
brushThickness = 25
eraserThickness = 100

# Load top images (toolbar)
folderPath = "E:\\Advance Computer Vision with Python\\main_en\\Project 5 AI Virtual Painter\\PainterImg"
myList = os.listdir(folderPath)  # Get the list of files in the folder
print(myList)
overlayList = [cv2.imread(f"{folderPath}/{imPath}") for imPath in myList]  # Read images
print(len(overlayList))

header = overlayList[0]  # Default select the first image
# print(header.shape) # h w c height width depth
# header = header[:152, :, :]
drawColor = (0, 0, 255)  # Default brush color

# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

detector = htm.handDetector(detectionCon=0.65, maxHands=1)  # Initialize hand detector
xp, yp = 0, 0  # Previous coordinate point
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # Create canvas
# np.zeros((720, 1280, 3), np.uint8) generates an array with 720 rows, 1280 columns, and 3 channels
# (720, 1280, 3) represents the height, width, and color channels (RGB) of the image
# np.uint8 specifies the data type as 8-bit unsigned integer, suitable for representing image pixel values (0-255)

modeChanged = False  # Initialize mode switch flag

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip image horizontally

    # 2. Detect hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    lmList = lmList[0]

    if len(lmList) != 0:
        # Get coordinates of index and middle fingertips
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            if not modeChanged:  # Check if mode has just switched
                xp, yp = 0, 0
                modeChanged = True  # Mark mode as switched

            print("Selection Mode")
            if y1 < 125:  # Check if fingers are in toolbar area
                if 0 < x1 < 200:
                    header = overlayList[0]
                    drawColor = (255, 0, 0)  # Blue
                elif 300 < x1 < 500:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)  # Red
                elif 600 < x1 < 800:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)  # Green
                elif 900 < x1 < 1100:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)  # Eraser
            cv2.rectangle(
                img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED
            )  # Draw selection box

        # 5. If drawing mode - only index finger is up
        if fingers[1] and not fingers[2]:
            if modeChanged:  # Check if mode has just switched
                xp, yp = x1, y1
                modeChanged = False  # Reset flag
                # Use a variable modeChanged to track if the mode has just switched
                # Reset the coordinates of the previous point each time switching from selection mode (two fingers up) to drawing mode (only one finger up)
                # x1, y1 are the current finger position coordinates
                # xp, yp are the coordinates of the previous finger position, used to store the finger position in the previous frame, to draw lines between the current frame and the previous frame
                # In drawing mode, the program will draw a line between xp, yp and x1, y1, allowing continuous line drawing as the finger moves
                # When the mode switches (e.g., from selection mode to drawing mode), xp and yp are reset to avoid unnecessary connections
                # Suppose I start from selection mode, modeChanged = False, it will execute xp, yp = 0, 0, and change modeChanged to True. After selection mode, it becomes drawing mode, if modeChanged is true, it will assign the current finger coordinates to xp, yp through xp, yp = x1, y1, then change modeChanged to False to reset the flag
                # If I don't assign x1, y1 to xp, yp, xp, yp would be the last coordinates from the previous drawing mode, because you'll find that all judgments in selection mode are based on x1, y1, it doesn't touch xp, yp
                # Then why don't I just set xp, yp to 0 after the selection mode ends, thinking about it, it seems to make sense, but I've written so much, don't want to delete QAQ

            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)  # Draw current point
            print("Drawing Mode")

            if xp == 0 and yp == 0:  # Initialize xp and yp
                xp, yp = x1, y1
                # In drawing mode, when first entering, xp and yp are 0, 0, which would cause drawing a line from the top-left corner of the canvas to the current finger position. By this initialization, assigning the current finger position x1, y1 to xp, yp ensures that line drawing only starts after the finger moves

            # Choose eraser or brush based on current color
            if drawColor == (0, 0, 0):  # Eraser mode
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:  # Brush mode
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # Process canvas image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # Convert imgCanvas to grayscale image, this is done to simplify subsequent threshold operations
    _, imgInv = cv2.threshold(imgGray, 25, 255, cv2.THRESH_BINARY_INV)
    # Apply threshold inversion to the grayscale image. Pixels below 25 become 255, above 25 become 0. This creates an inverted binary image
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # Convert the inverted grayscale image back to a three-channel image for bitwise operations with color images
    img = cv2.bitwise_and(img, imgInv)
    # Perform bitwise AND operation between the camera image and inverted image. This removes the parts already drawn on the canvas from the camera image
    img = cv2.bitwise_or(img, imgCanvas)
    # Perform bitwise OR operation between the processed camera image and canvas image. This overlays the drawn content onto the camera image, forming the final composite image
    # This is a designed process, aiming to composite the drawn content with the camera image, each step has its specific purpose:
    # 1. Grayscale conversion: Simplify image data for threshold processing
    # 2. Threshold inversion: Create a mask to highlight drawn content
    # 3. Color conversion: Ensure mask matches original image channels
    # 4. Bitwise AND operation: Remove drawn parts from camera image
    # 5. Bitwise OR operation: Overlay drawn content onto camera image
    # This process takes img (real-time camera captured image) and imgCanvas (current drawn content, i.e., canvas) as input, and outputs img (composited image, including camera image and drawn content)

    # Set top toolbar image
    header = cv2.resize(header, (1280, 125))  # Adjust toolbar size
    img[0:125, 0:1280] = header  # Place toolbar at the top

    cv2.imshow(
        "Image", img
    )  # Display composited image window, including camera image and drawn content
    cv2.imshow(
        "Canvas", imgCanvas
    )  # Display current drawing canvas, only including drawn content
    cv2.imshow(
        "Inv", imgInv
    )  # Display inverted binary image, for debugging and viewing mask effect
    cv2.waitKey(1)
