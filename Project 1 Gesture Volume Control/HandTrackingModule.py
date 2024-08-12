"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone

Modified by: Diraw
Date: 20240811
Description:
1. Modified the initialization of the `Hands` object to use named parameters for better clarity and compatibility with the latest version of the mediapipe library. This change ensures that the parameters are correctly mapped to the expected arguments in the `Hands` class.
2. Added a line to flip the image horizontally using `cv2.flip(img, 1)` to ensure the hand movements appear mirrored, which is more intuitive for user interaction.
3. Added the code lmList = lmList[0] at line 59 in VolumeHandControl.py to fix the error: IndexError: tuple index out of range.
"""

import cv2
import mediapipe as mp
import time
import math


# 手部检测类
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # 初始化参数
        self.mode = mode  # 是否静态模式
        self.maxHands = maxHands  # 最大检测手数
        self.detectionCon = detectionCon  # 检测置信度阈值
        self.trackCon = trackCon  # 跟踪置信度阈值

        # 初始化手部检测模块
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

        self.mpDraw = mp.solutions.drawing_utils  # 绘制工具
        self.tipIds = [4, 8, 12, 16, 20]  # 指尖的id

    # 查找手部并绘制
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
        self.results = self.hands.process(imgRGB)  # 处理图像

        if self.results.multi_hand_landmarks:  # 如果检测到手
            for handLms in self.results.multi_hand_landmarks:
                if draw:  # 是否绘制
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    # 查找手部位置
    def findPosition(self, img, handNo=0, draw=True):
        xList = []  # 存储x坐标
        yList = []  # 存储y坐标
        bbox = []  # 边界框
        self.lmList = []  # 存储手部关键点

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  # 选择第handNo只手
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # 获取图像尺寸
                cx, cy = int(lm.x * w), int(lm.y * h)  # 转换为像素坐标
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])  # 添加到列表
                if draw:  # 是否绘制
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)  # 计算边界框
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:  # 绘制边界框
                cv2.rectangle(
                    img,
                    (bbox[0] - 20, bbox[1] - 20),
                    (bbox[2] + 20, bbox[3] + 20),
                    (0, 255, 0),
                    2,
                )

        return self.lmList, bbox

    # 检测手指是否伸直
    def fingersUp(self):
        # 在OpenCV中，图像的左上角是坐标系的原点(0,0)，x坐标向右增加，y坐标向下增加
        # 如果使用了 img = cv2.flip(img, 1)，图像会水平翻转。这样一来，x坐标的大小关系就会反转：图像的右侧变成了坐标系的左侧。因此，x坐标越往左越大。
        # 再理性分析一波：
        # “在OpenCV中，图像的左上角是坐标系的原点(0,0)，x坐标向右增加，y坐标向下增加”这句话是永远不变的，也就是说，我们看电脑屏幕上面反馈的图像，永远是“左上角是坐标系的原点(0,0)，x坐标向右增加，y坐标向下增加”
        # 在翻转前，我们的动作和电脑显示的画面是左右相反的，也就是此时在我们的视角里，越往右，电脑中图像的x坐标越小；如果我们翻转图像之后，在我们的视角里，越往左，电脑中图像的x坐标越小，此时我们视角中的坐标系和电脑显示的坐标系就是一样的了，都是越往左，x坐标越小
        # 在此基础上，判断右手大拇指伸直的条件就是，4的x坐标小于3的x坐标

        fingers = []
        # 大拇指
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            # self.tipIds = [4, 8, 12, 16, 20]  # 指尖的id
            # self.tipIds[0] 是大拇指指尖的索引(4)，self.tipIds[0] - 1 是大拇指指尖前一个关节的索引(3)
            # [1] 从 self.lmList 中获取该关节的 x 坐标，因为self.lmList.append([id, cx, cy])，第0维是id，第1维才是x坐标
            # 此时这个条件适用于右手大拇指伸直的情况
            # 这种判断方式真的很难评，它并不明确区分左右手，而是基于位置关系进行推断。真正的左右手区分需要使用 mediapipe 提供的 multi_handedness 属性
            # multi_handedness 是 MediaPipe 提供的一个属性，用于识别检测到的手是左手还是右手。它包含关于手的置信度和标签（"Left" 或 "Right"）的信息
            fingers.append(1)  # 右手大拇指伸直的话，返回1
        else:
            fingers.append(0)  # 右手大拇指弯曲的话，返回0
        # 其他四根手指
        for id in range(1, 5):  # 遍历 4 个 id：1, 2, 3, 4
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                # 这里是判断其他手指，在OpenCV中，y坐标向下增加
                fingers.append(1)  # 手指伸直
            else:
                fingers.append(0)  # 手指弯曲
        return fingers

    # 计算两点之间的距离
    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]  # 第一个点
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]  # 第二个点
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 中点

        if draw:  # 是否绘制
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)  # 计算距离，已知两个直角边求斜边
        return length, img, [x1, y1, x2, y2, cx, cy]


# 主函数
def main():
    pTime = 0  # 上一帧时间
    cap = cv2.VideoCapture(0)  # 打开摄像头
    detector = handDetector()  # 初始化检测器

    while True:
        success, img = cap.read()  # 读取图像
        img = cv2.flip(img, 1)  # 水平翻转图像
        img = detector.findHands(img)  # 检测手部
        lmList, bbox = detector.findPosition(img)  # 获取位置
        if len(lmList) != 0:
            print(lmList[4])  # 打印指定关键点

        cTime = time.time()  # 当前时间
        fps = 1 / (cTime - pTime)  # 计算帧率
        pTime = cTime

        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )  # 显示帧率

        cv2.imshow("Image", img)  # 显示图像
        cv2.waitKey(1)  # 等待按键


if __name__ == "__main__":
    main()  # 运行主函数
