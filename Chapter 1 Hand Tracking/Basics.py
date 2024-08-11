import cv2
import mediapipe as mp
import time

# Open the camera
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Initialize the hand detection module
mpHands = mp.solutions.hands  # Reference MediaPipe's hand solution module
hands = (
    mpHands.Hands()
)  # Create a Hands object for detecting and tracking hand landmarks
mpDraw = (
    mp.solutions.drawing_utils
)  # Reference drawing tools for drawing detected hand landmarks and connections

# Initialize time variables for calculating FPS
pTime = 0  # Previous time for the previous frame
cTime = 0  # Current time for the current frame
# cTime - pTime calculates the time difference to compute FPS. Finally, cTime is assigned to pTime for use in the next loop

while True:
    # Read the camera image
    success, img = cap.read()
    # success: Boolean indicating if the frame was read successfully
    # img: The image frame read; this value may be empty if reading fails

    # Flip the image horizontally
    img = cv2.flip(img, 1)
    # Dimension 0 represents the vertical direction (height), corresponding to the number of rows in the image, top to bottom
    # Dimension 1 represents the horizontal direction (width), corresponding to the number of columns in the image, left to right

    # Convert the image from BGR format to RGB format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # BGR is the default color format in OpenCV, representing Blue, Green, and Red. This order is opposite to the commonly used RGB (Red, Green, Blue). Conversion to RGB is because many image processing libraries (like MediaPipe) use this format for processing

    # Process the image to detect hands
    results = hands.process(imgRGB)

    # If hands are detected
    if results.multi_hand_landmarks:
        # Iterate over each detected hand
        for handLms in results.multi_hand_landmarks:
            # results.multi_hand_landmarks returns a list containing landmark information for each detected hand. If multiple hands are detected, it contains multiple elements, each representing all landmarks of one hand

            # Iterate over hand landmarks
            for id, lm in enumerate(handLms.landmark):
                # enumerate returns an iterator, each iteration returns a tuple containing the index and value
                # id is the index of the hand landmark, lm is short for landmark, representing the coordinate information of the hand landmark

                # Get the dimensions of the image
                h, w, c = img.shape
                # img.shape returns a tuple containing the image dimensions: height (number of rows), width (number of columns), and number of channels (e.g., 3 for RGB images)

                # Calculate the coordinates of the landmark in the image
                cx, cy = int(lm.x * w), int(lm.y * h)
                # lm.x and lm.y are the normalized coordinates of the landmark, ranging from 0 to 1. By multiplying by the image width and height, they can be converted to pixel coordinates in the image

                print(id, cx, cy)

                # Draw a circle at the landmark
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), -1)
                # img is where to draw the image, (cx, cy) is the center of the circle, 15 is the radius
                # (255, 0, 255) is the color of the circle (BGR format), which is purple here
                # Red: (0, 0, 255), Green: (0, 255, 0), Blue: (255, 0, 0), Yellow: (0, 255, 255), Cyan: (255, 255, 0), Magenta: (255, 0, 255), White: (255, 255, 255), Black: (0, 0, 0)
                # cv2.FILLED or -1 for a filled circle, or a specific number for border thickness

            # Draw hand landmarks and connections
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # img is the image to draw on, handLms are the coordinates of the hand landmarks
            # mpHands.HAND_CONNECTIONS defines the connections between hand landmarks for drawing the skeleton structure

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the image
    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
    )
    # img is where to draw the text, str(int(fps)) is the text content to display, which is the integer part of FPS
    # (10, 70) is the bottom-left corner of the text, cv2.FONT_HERSHEY_PLAIN is the font style
    # 3 is the font size, (255, 0, 255) is the text color (purple, BGR format), 3 is the thickness of the text
    # cv2.putText does not support keyword arguments; parameters must be provided in order

    # Show the image
    cv2.imshow("Image", img)  # Display the image in a window titled "Image"
    cv2.waitKey(
        1
    )  # Wait for a keyboard event; a parameter of 1 means to wait for 1 millisecond
    # This also allows the image window to respond to user input (such as closing the window)

    # Detect exit key
    if cv2.waitKey(1) & 0xFF == ord(
        "q"
    ):  # ord('q') gets the ASCII value of the character 'q'
        # cv2.waitKey(1) & 0xFF is used to read keyboard input
        # cv2.waitKey(1) returns a 32-bit integer, where the lower 8 bits are the actual key value, and & 0xFF is a bitwise operation to extract these 8 bits
        # "Lower 8 bits" refers to the rightmost 8 bits in the binary representation of a number. These bits represent the smaller portion of the value, as opposed to the "higher 8 bits" (leftmost 8 bits), which represent the larger portion. For a 32-bit integer, the lower 8 bits are used to represent the actual key value from keyboard input
        break

cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows
