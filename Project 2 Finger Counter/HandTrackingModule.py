"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone

Modified by: Diraw
Date: 20240812
Description:
1. Modified the initialization of the `Hands` object to use named parameters for better clarity and compatibility with the latest version of the mediapipe library. This change ensures that the parameters are correctly mapped to the expected arguments in the `Hands` class.
2. Added a line to flip the image horizontally using `cv2.flip(img, 1)` to ensure the hand movements appear mirrored, which is more intuitive for user interaction
"""

import cv2
import mediapipe as mp
import time


# 手部检测器类
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # 初始化参数
        self.mode = mode  # 是否为静态模式
        self.maxHands = maxHands  # 最大检测手数
        self.detectionCon = detectionCon  # 检测置信度
        self.trackCon = trackCon  # 跟踪置信度

        # 初始化MediaPipe手部模型
        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(
        #     self.mode, self.maxHands, self.detectionCon, self.trackCon
        # )
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )

        # 初始化绘图工具
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # 将图像转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 处理图像，检测手部
        self.results = self.hands.process(imgRGB)

        # 如果检测到手部
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # 绘制手部关键点和连接
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []  # 存储手部关键点位置
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # 获取图像尺寸
                h, w, c = img.shape
                # 计算关键点的像素位置
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                # 绘制关键点
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList


# 主函数
def main():
    pTime = 0  # 上一帧时间
    cTime = 0  # 当前帧时间
    cap = cv2.VideoCapture(0)  # 打开摄像头
    detector = handDetector()  # 创建手部检测器对象

    while True:
        success, img = cap.read()  # 读取摄像头帧
        img = cv2.flip(img, 1)  # 水平翻转图像
        img = detector.findHands(img)  # 检测手并绘制
        lmList = detector.findPosition(img)  # 获取手部关键点位置
        if len(lmList) != 0:
            print(lmList[4])  # 打印拇指尖的坐标

        cTime = time.time()  # 获取当前时间
        fps = 1 / (cTime - pTime)  # 计算帧率
        pTime = cTime  # 更新上一帧时间

        # 在图像上显示帧率
        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        # 显示图像
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
