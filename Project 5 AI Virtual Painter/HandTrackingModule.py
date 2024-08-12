"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone

Modified by: Diraw
Date: 20240812
Description:
1. Modified the initialization of the `Hands` object to use named parameters for better clarity and compatibility with the latest version of the mediapipe library. This change ensures that the parameters are correctly mapped to the expected arguments in the `Hands` class.
2. Added a line to flip the image horizontally using `cv2.flip(img, 1)` to ensure the hand movements appear mirrored, which is more intuitive for user interaction
3. Added the code lmList = lmList[0] at line 59 in VolumeHandControl.py to fix the error: IndexError: tuple index out of range.
4. Set the top toolbar image by resizing it with cv2.resize(header, (1280, 125)) and placing it at the top of the frame with img[0:125, 0:1280] = header to ensure it fits the toolbar area.
5. Added a mode change flag to ensure proper initialization of drawing coordinates. This prevents unwanted lines when switching from selection to drawing mode by resetting xp and yp to the current finger position upon mode change.
6. Adjusted the threshold value in cv2.threshold from 50 to 25 to improve the visibility of blue lines by ensuring they are correctly captured in the binary inversion process.
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # 初始化手部检测器参数
        self.mode = mode  # 是否为静态模式
        self.maxHands = maxHands  # 最大检测手的数量
        self.detectionCon = detectionCon  # 检测置信度
        self.trackCon = trackCon  # 跟踪置信度

        # 初始化 Mediapipe 手部模型
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
        self.mpDraw = mp.solutions.drawing_utils  # 用于绘制手部关键点
        self.tipIds = [4, 8, 12, 16, 20]  # 手指尖端的关键点 ID

    def findHands(self, img, draw=True):
        # 检测手部并绘制关键点
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        self.results = self.hands.process(imgRGB)  # 处理图像

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # 绘制手部关键点及连接
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        return img

    def findPosition(self, img, handNo=0, draw=True):
        # 获取手部关键点坐标
        xList = []
        yList = []
        bbox = []
        self.lmList = []  # 存储关键点信息
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # 转换为像素坐标
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])  # 添加到列表
                if draw:
                    # 绘制每个关键点
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        if xList and yList:  # 检查列表是否不为空
            # 计算边界框
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                # 绘制边界框
                cv2.rectangle(
                    img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2
                )

        return self.lmList, bbox

    def fingersUp(self):
        # 判断哪些手指竖起
        fingers = []
        # 拇指
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 其他手指
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        # 计算两个关键点之间的距离
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # 绘制距离线和中心点
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)  # 计算欧几里得距离

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # 主函数，用于测试模块
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # 打开摄像头
    detector = handDetector()  # 创建手部检测器对象
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # 水平翻转图像
        img = detector.findHands(img)  # 检测手部
        lmList, bbox = detector.findPosition(img)  # 获取关键点坐标
        if len(lmList) != 0:
            print(lmList[4])  # 打印拇指的坐标

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # 计算帧率
        pTime = cTime

        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )  # 显示帧率

        cv2.imshow("Image", img)  # 显示图像
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
