import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100  # 边框缩减
smoothening = 7  # 平滑系数
#########################
# 以上是用来在代码中创建视觉分隔线的注释。它们可以帮助开发者更清晰地分隔代码的不同部分，使代码更易读。这些行本身并没有功能，只是为了提高代码的可读性

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()  # 获取屏幕的宽度和高度
# autopy.screen.size() 返回屏幕的尺寸，以像素为单位
# wScr 是屏幕的宽度，hScr 是屏幕的高度
# print(wScr, hScr)

while True:
    # 1. 检测手部标志点
    success, img = cap.read()
    img = cv2.flip(img, 1)  # 水平翻转图像
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. 获取食指和中指指尖的位置
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. 检查哪些手指是抬起的
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(
            img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2
        )

        # 4. 只有食指抬起：移动模式
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. 转换坐标
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # np.interp() 是一个用于线性插值的函数。它的基本用法: np.interp(x, xp, fp)
            # 其中，x 为需要插值的点，xp 为输入数据点的范围（已知的 x 值），fp 为输出数据点的范围（已知的 y 值）
            # 在这里，x1 和 y1 是手指在摄像头图像中的位置，(frameR, wCam - frameR) 是摄像头图像中手指活动的范围，(0, wScr) 和 (0, hScr) 是屏幕的坐标范围
            # 通过 np.interp()，程序将手指在摄像头图像中的位置映射到屏幕的坐标上，从而实现手指移动与鼠标光标移动的对应关系。这使得手指在摄像头中的移动能够控制屏幕上的鼠标光标

            # 6. 平滑处理数值
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # clocX 和 clocY 是当前平滑处理后的坐标，plocX 和 plocY 是前一帧的坐标
            # x3 和 y3 是当前帧经过插值后的目标坐标，smoothening 是一个平滑系数，用于控制平滑的程度
            # 每次更新时，当前位置（clocX, clocY）向目标位置（x3, y3）移动一小部分，smoothening 控制移动的步长，值越大，移动越慢，越平滑
            # 平滑处理会让鼠标指针移动稍微慢一些，但这样做的目的是为了减少抖动，使移动更加平稳
            # 如果你觉得移动速度太慢，可以尝试减小 smoothening 的值，这样指针会更快地跟随手指移动

            # 7. 移动鼠标
            # autopy.mouse.move(wScr - clocX, clocY)
            # 如果没有翻转，则使用上面这条语句
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. 食指和中指都抬起：点击模式
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. 计算两指间的距离
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)

            # 10. 如果距离很短，则点击鼠标
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. 帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. 显示
    cv2.imshow("Image", img)
    cv2.waitKey(1)
