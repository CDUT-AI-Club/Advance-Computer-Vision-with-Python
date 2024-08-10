# cv2.__version__ = 4.10.0, mp.__version__ = 0.10.14

import cv2
import mediapipe as mp
import time


# 定义手部检测类
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # 初始化参数
        self.mode = mode  # 静态图像模式
        self.maxHands = maxHands  # 最大检测手数
        self.detectionCon = detectionCon  # 检测置信度
        self.trackCon = trackCon  # 跟踪置信度
        self.mpHands = mp.solutions.hands  # Mediapipe手部解决方案
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils  # 用于绘制手部连接

    def findHands(self, img, draw=True):
        # 将图像从BGR转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 处理图像，检测手部
        self.results = self.hands.process(imgRGB)
        # 如果检测到手部
        if self.results.multi_hand_landmarks:
            # 遍历每个手部
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # 绘制手部连接
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        # 初始化列表存储手部位置
        lmList = []
        # 如果检测到手部
        if self.results.multi_hand_landmarks:
            # self.results 是在 findHands 方法中定义的。由于 findHands 方法在 findPosition 方法之前被调用，因此 self.results 会被正确地初始化并存储检测结果
            # 这种设计依赖于调用顺序，我们得确保在调用 findPosition 之前已经调用过 findHands，否则 self.results 可能没有数据，导致 findPosition 无法正常工作
            # 在 Python 中，self 参数用于引用类的实例。只要在类的方法中通过 self 定义了属性（例如 self.results），该属性就可以在同一个类的其他方法中访问和使用。这样可以在不同的方法之间共享数据

            # 获取指定手部
            myHand = self.results.multi_hand_landmarks[handNo]
            # 遍历每个关键点
            for id, lm in enumerate(myHand.landmark):
                # 获取图像尺寸
                h, w, c = img.shape
                # 计算关键点在图像中的位置
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    # 在图像上绘制关键点
                    cv2.circle(img, (cx, cy), 1, (255, 0, 255), -1)
                    # 由于现有的 findHands，再有的 handDetector，所以“关键点”的图层在“手部连接”之上，前一个程序则相反

        return lmList


# 主函数
def main():
    pTime = 0  # 前一帧时间
    cTime = 0  # 当前时间
    cap = cv2.VideoCapture(0)  # 打开摄像头
    detector = handDetector()  # 创建手部检测器
    while True:
        success, img = cap.read()  # 读取摄像头图像
        img = cv2.flip(img, 1)  # 水平翻转图像
        img = detector.findHands(img)  # 检测手部
        lmList = detector.findPosition(img)  # 获取手部关键点位置
        if len(lmList) != 0:
            print(lmList[4])  # 打印大拇指指尖位置
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # 计算帧率
        pTime = cTime
        # 在图像上显示帧率
        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )
        cv2.imshow("Image", img)  # 显示图像
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # 按下'q'退出


if __name__ == "__main__":
    main()
