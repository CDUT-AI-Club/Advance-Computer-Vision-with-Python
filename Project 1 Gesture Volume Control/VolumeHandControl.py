import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# from pprint import pprint

# 设置摄像头的宽度和高度
wCam, hCam = 640, 480

# 初始化摄像头
cap = cv2.VideoCapture(0)
# if cap.isOpened():
#     print("Camera successfully opened")
# else:
#     print("Failed to open camera")
cap.set(3, wCam)  # 设置宽度
cap.set(4, hCam)  # 设置高度
# 设置摄像头捕获的图像分辨率为 640x480，即有 640x480 个像素点

pTime = 0  # 前一帧的时间

# 初始化手部检测器，设置检测置信度
detector = htm.handDetector(detectionCon=0.7)

# 获取音频设备
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# 获取音量范围
volRange = volume.GetVolumeRange()
minVol = volRange[0]  # 最小音量
maxVol = volRange[1]  # 最大音量
# print(volRange)  # (-45.0, 0.0, 1.0)

vol = 0  # 当前音量
volBar = 400  # 音量条位置
volPer = 0  # 音量百分比

while True:
    success, img = cap.read()  # 读取摄像头图像
    # if success:
    #     print("Image captured successfully")
    #     # cv2.imshow("Captured Image", img)
    #     # cv2.waitKey(0)
    # else:
    #     print("Failed to capture image")
    img = cv2.flip(img, 1)  # 水平翻转图像
    img = detector.findHands(img)  # 检测手部
    lmList = detector.findPosition(img, draw=False)  # 获取手部关键点列表
    # print(type(lmList))
    # pprint(lmList)
    # lmList_is_empty = all(len(lst) == 0 for lst in lmList)
    lmList = lmList[0]
    # pprint(lmList)

    if len(lmList) != 0:
        # 获取食指和拇指的坐标
        try:
            if len(lmList) > 4 and len(lmList[4]) > 1:
                x1, y1 = lmList[4][1], lmList[4][2]
            else:
                raise IndexError("Landmark index out of range")
        except IndexError as e:
            print(e)
        x1, y1 = lmList[4][1], lmList[4][2]  # 拇指
        x2, y2 = lmList[8][1], lmList[8][2]  # 食指
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 中心点
        # print(1)

        # 绘制圆和线
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # 拇指圆
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)  # 食指圆
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # 连接线
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # 中心点圆

        # 计算两点之间的距离
        length = math.hypot(x2 - x1, y2 - y1)

        # 将距离转换为音量
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        print(int(length), vol)

        # 设置音量
        volume.SetMasterVolumeLevel(vol, None)

        # 如果距离小于50，改变圆的颜色
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # 绘制音量条
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)  # 外框
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)  # 填充
    cv2.putText(
        img, f"{int(volPer)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3
    )

    # 计算帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # 显示帧率
    cv2.putText(
        img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3
    )

    # 显示图像
    cv2.imshow("Img", img)
    cv2.waitKey(1)
