import cv2
import mediapipe as mp
import time


# Define hand detection class
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialize parameters
        self.mode = mode  # Static image mode
        self.maxHands = maxHands  # Maximum number of hands to detect
        self.detectionCon = detectionCon  # Detection confidence
        self.trackCon = trackCon  # Tracking confidence
        self.mpHands = mp.solutions.hands  # Mediapipe hand solution
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils  # For drawing hand connections

    def findHands(self, img, draw=True):
        # Convert image from BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process image to detect hands
        self.results = self.hands.process(imgRGB)
        # If hands are detected
        if self.results.multi_hand_landmarks:
            # Iterate over each hand
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw hand connections
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        # Initialize list to store hand positions
        lmList = []
        # If hands are detected
        if self.results.multi_hand_landmarks:
            # self.results is defined in the findHands method. Since findHands is called before findPosition, self.results will be properly initialized and store detection results.
            # This design relies on the call order, so we must ensure findHands is called before findPosition; otherwise, self.results may lack data, causing findPosition to malfunction.
            # In Python, the self parameter refers to the class instance. As long as an attribute (e.g., self.results) is defined via self in a class method, it can be accessed and used in other methods of the same class, allowing data sharing between methods.

            # Get the specified hand
            myHand = self.results.multi_hand_landmarks[handNo]
            # Iterate over each landmark
            for id, lm in enumerate(myHand.landmark):
                # Get image dimensions
                h, w, c = img.shape
                # Calculate landmark position in the image
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    # Draw the landmark on the image
                    cv2.circle(img, (cx, cy), 1, (255, 0, 255), -1)
                    # Due to findHands and handDetector, the "landmark" layer is above the "hand connections," unlike the previous program.

        return lmList


# Main function
def main():
    pTime = 0  # Previous frame time
    cTime = 0  # Current time
    cap = cv2.VideoCapture(0)  # Open the camera
    detector = handDetector()  # Create a hand detector
    while True:
        success, img = cap.read()  # Read camera image
        img = cv2.flip(img, 1)  # Flip image horizontally
        img = detector.findHands(img)  # Detect hands
        lmList = detector.findPosition(img)  # Get hand landmark positions
        if len(lmList) != 0:
            print(lmList[4])  # Print thumb tip position
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Calculate frame rate
        pTime = cTime
        # Display frame rate on the image
        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )
        cv2.imshow("Image", img)  # Show image
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Press 'q' to exit


if __name__ == "__main__":
    main()
