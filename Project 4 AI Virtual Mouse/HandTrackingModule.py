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
3. Updated the findPosition method to check if xList and yList are not empty before calculating the minimum and maximum values. This prevents errors when no hand landmarks are detected.
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # 初始化手部检测器
        self.mode = mode  # 模式：静态或动态
        self.maxHands = maxHands  # 最大检测手数
        self.detectionCon = detectionCon  # 检测置信度
        self.trackCon = trackCon  # 跟踪置信度

        self.mpHands = mp.solutions.hands  # Mediapipe手部解决方案
        # self.hands = self.mpHands.Hands(
        #     self.mode, self.maxHands, self.detectionCon, self.trackCon
        # )
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils  # 绘图工具
        self.tipIds = [4, 8, 12, 16, 20]  # 指尖的ID

    def findHands(self, img, draw=True):
        # 查找手部并绘制
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
        self.results = self.hands.process(imgRGB)  # 处理图像

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )  # 绘制手部连接

        return img

    def findPosition(self, img, handNo=0, draw=True):
        # 查找手部位置
        xList = []  # X坐标列表
        yList = []  # Y坐标列表
        bbox = []  # 边界框
        self.lmList = []  # 存储手部关键点

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # 获取图像尺寸
                cx, cy = int(lm.x * w), int(lm.y * h)  # 转换为像素坐标
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # 绘制关键点

        if xList and yList:  # 检查列表是否不为空
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax  # 计算边界框

            if draw:
                cv2.rectangle(
                    img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2
                )  # 绘制边界框

        return self.lmList, bbox

    def fingersUp(self):
        # 判断手指是否竖起
        fingers = []
        # 拇指
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            # 同前几个Project，翻转前判断右手是if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            # 翻转后，变大于为小于号
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
        # 计算两点之间的距离
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 中点坐标

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)  # 绘制线条
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)  # 绘制起点
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)  # 绘制终点
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)  # 绘制中点
            length = math.hypot(x2 - x1, y2 - y1)  # 计算距离

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # 主函数
    pTime = 0  # 上一帧时间
    cTime = 0  # 当前时间
    cap = cv2.VideoCapture(0)  # 打开摄像头
    detector = handDetector()  # 创建手部检测器

    while True:
        success, img = cap.read()  # 读取摄像头图像
        img = cv2.flip(img, 1)  # 水平翻转图像
        img = detector.findHands(img)  # 查找手部
        lmList, bbox = detector.findPosition(img)  # 获取手部位置
        if len(lmList) != 0:
            print(lmList[4])  # 打印大拇指位置

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # 计算帧率
        pTime = cTime

        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )  # 显示帧率

        cv2.imshow("Image", img)  # 显示图像
        cv2.waitKey(1)  # 等待按键


if __name__ == "__main__":
    main()  # 运行主函数
