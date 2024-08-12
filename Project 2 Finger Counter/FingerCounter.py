# 此脚本只能检测右手的手势
import cv2
import time
import os
import HandTrackingModule as htm

# 设置摄像头的宽度和高度，即分辨率
wCam, hCam = 640, 480

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # 设置摄像头宽度
cap.set(4, hCam)  # 设置摄像头高度

# 指定手指图像文件夹路径
folderPath = "E:\\Advance Computer Vision with Python\\main\\Project 2 Finger Counter\\FingerImages"
myList = os.listdir(folderPath)  # 获取文件夹中的所有文件名列表
print(myList)

overlayList = []  # 用于存储手指图像的列表
for imPath in myList:
    # 读取每张手指图像
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)  # 将图像添加到列表中

print(len(overlayList))  # 输出图像数量

pTime = 0  # 初始化上一帧时间

# 创建手部检测器对象，设置检测置信度为0.75
detector = htm.handDetector(detectionCon=0.75)

# 手指尖的ID列表
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()  # 从摄像头读取帧
    img = cv2.flip(img, 1)  # 水平翻转图像
    img = detector.findHands(img)  # 检测手并绘制手部关键点
    lmList = detector.findPosition(img, draw=False)  # 获取手部关键点位置列表

    if len(lmList) != 0:
        fingers = []

        # 检测拇指（根据拇指尖和拇指第二关节的x坐标判断）
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            # 如果按照源代码来的话if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]判断的是未翻转前的右手大拇指的伸直
            # 所以如果我选择了img = cv2.flip(img, 1)，此时if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]判断的是水平翻转后前的左手大拇指的伸直，要保持检测的是右手的话，遂变大于为小于
            # 故此脚本只能检测右手的手势
            fingers.append(1)  # 1表示该手指是伸出的
        else:
            fingers.append(0)  # 0表示该手指是弯曲的

        # 检测其他四根手指（根据手指尖和指根的y坐标判断）
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)  # 计算伸出的手指数量
        print(totalFingers)

        # 选择对应手指数量的图像进行覆盖显示
        h, w, c = overlayList[totalFingers - 1].shape
        # totalFingers - 1 是因为列表索引是从 0 开始的，而 totalFingers 表示手指的数量（从 1 开始）。所以要减 1 来正确访问列表中对应的图像
        # 因此把0放在了列表的最后，因为此时没有手指伸直，totalFingers为0，减1得到-1，就是列表的最后一个了
        img[0:h, 0:w] = overlayList[totalFingers - 1] # 将对应数量手指的图像放置在摄像头图像的左上角

        # 绘制矩形并显示手指数量
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            img,
            str(totalFingers),
            (45, 375),
            cv2.FONT_HERSHEY_PLAIN,
            10,
            (255, 0, 0),
            25,
        )

    cTime = time.time()  # 获取当前帧时间
    fps = 1 / (cTime - pTime)  # 计算帧率
    pTime = cTime  # 更新上一帧时间

    # 在图像上显示帧率
    cv2.putText(
        img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    # 显示处理后的图像
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # 等待按键，1毫秒
