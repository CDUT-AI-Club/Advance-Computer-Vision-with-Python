import cv2
import numpy as np
import os
import HandTrackingModule as htm

# 设置画笔和橡皮擦的粗细
brushThickness = 25
eraserThickness = 100

# 加载顶部图像（工具栏）
folderPath = "E:\\Advance Computer Vision with Python\\main\\Project 5 AI Virtual Painter\\PainterImg"
myList = os.listdir(folderPath)  # 获取文件夹中的文件列表
print(myList)
overlayList = [cv2.imread(f"{folderPath}/{imPath}") for imPath in myList]  # 读取图像
print(len(overlayList))

header = overlayList[0]  # 默认选择第一个图像
# print(header.shape) # h w c 高度 宽度 深度
# header = header[:152, :, :]
drawColor = (0, 0, 255)  # 默认画笔颜色

# 设置视频捕获
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # 设置宽度
cap.set(4, 720)  # 设置高度

detector = htm.handDetector(detectionCon=0.65, maxHands=1)  # 初始化手部检测器
xp, yp = 0, 0  # 上一个坐标点
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # 创建画布
# np.zeros((720, 1280, 3), np.uint8) 生成一个 720 行、1280 列、3 通道的数组
# (720, 1280, 3) 表示图像的高度、宽度和颜色通道（RGB）
# np.uint8 指定数据类型为 8 位无符号整数，适合表示图像像素值（0-255）

modeChanged = False  # 初始化模式切换标志

while True:
    # 1. 导入图像
    success, img = cap.read()
    img = cv2.flip(img, 1)  # 水平翻转图像

    # 2. 检测手部关键点
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    lmList = lmList[0]

    if len(lmList) != 0:
        # 获取食指和中指尖端的坐标
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. 检查哪些手指竖起
        fingers = detector.fingersUp()

        # 4. 如果是选择模式——两根手指竖起
        if fingers[1] and fingers[2]:
            if not modeChanged:  # 检查模式是否刚刚切换
                xp, yp = 0, 0
                modeChanged = True  # 标记模式已切换

            print("Selection Mode")
            if y1 < 125:  # 检查手指是否在工具栏区域
                if 0 < x1 < 200:
                    header = overlayList[0]
                    drawColor = (255, 0, 0)  # 蓝色
                elif 300 < x1 < 500:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)  # 红色
                elif 600 < x1 < 800:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)  # 绿色
                elif 900 < x1 < 1100:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)  # 橡皮擦
            cv2.rectangle(
                img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED
            )  # 绘制选择框

        # 5. 如果是绘画模式——只有食指竖起
        if fingers[1] and not fingers[2]:
            if modeChanged:  # 检查模式是否刚刚切换
                xp, yp = x1, y1
                modeChanged = False  # 重置标志
                # 使用一个变量 modeChanged 来跟踪模式是否刚刚切换
                # 每次从选择模式（两根手指竖起）切换到绘画模式（只有一根手指竖起）时，重置上一个点的坐标
                # x1, y1 是当前手指所在的位置坐标
                # xp, yp 是上一个手指位置的坐标，用于存储上一帧的手指位置，以便在当前帧和上一帧之间绘制线条
                # 在绘画模式下，程序会在 xp, yp 和 x1, y1 之间画线，这样可以在手指移动时绘制连续的线条
                # 当模式切换时（例如从选择模式到绘画模式），xp 和 yp 会被重置，以避免不必要的连线
                # 假设我从选择模式开始，modeChanged = False，将会执行xp, yp = 0, 0，并且将modeChanged改为True，选择模式之后就是变成绘画模式，if modeChanged判断成功，通过xp, yp = x1, y1将现在手指的坐标赋值给xp, yp，之后将modeChanged改为False重置标志
                # 如果我不把x1, y1赋值给xp, yp，xp, yp就是我上一个绘画模式最后的坐标，因为你会发现，选择模式里面的判断都是根据x1, y1，他没动xp, yp
                # 那我直接在选择模式结束之后令xp, yp等于0不就好了，这么一想，好像是这个道理，但是打了好多字，不想删QAQ

            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)  # 绘制当前点
            print("Drawing Mode")

            if xp == 0 and yp == 0:  # 初始化 xp 和 yp
                xp, yp = x1, y1
                # 在绘画模式下，第一次进入时 xp 和 yp 是 0, 0，这会导致从画布左上角画线到当前手指位置，通过这种初始化，将当前手指位置 x1, y1 赋值给 xp, yp，确保只有在手指移动后才开始绘制线条
            
            # 根据当前颜色选择橡皮擦或画笔
            if drawColor == (0, 0, 0):  # 橡皮擦模式
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:  # 画笔模式
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # 处理画布图像
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # 将 imgCanvas 转换为灰度图像，这样做是为了简化后续的阈值操作
    _, imgInv = cv2.threshold(imgGray, 25, 255, cv2.THRESH_BINARY_INV)
    # 对灰度图像应用阈值反转。低于 25 的像素变为 255，高于 25 的变为 0。这创建了一个反转的二值图像
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # 将反转后的灰度图像转换回三通道图像，以便与彩色图像进行按位操作
    img = cv2.bitwise_and(img, imgInv)
    # 将摄像头图像与反转图像进行按位与操作。这会移除摄像头图像中画布上已经绘制的部分
    img = cv2.bitwise_or(img, imgCanvas)
    # 将处理后的摄像头图像与画布图像进行按位或操作。这会将画布上的绘制内容叠加到摄像头图像上，形成最终的合成图像
    # 这是设计好的流程，目的是将绘制的内容与摄像头图像合成，每一步都有其特定的作用：
    # 1、灰度转换：简化图像数据，便于阈值处理
    # 2、阈值反转：创建一个掩膜，突出绘制内容
    # 3、颜色转换：确保掩膜与原始图像通道匹配
    # 4、按位与操作：移除摄像头图像中绘制的部分
    # 5、按位或操作：将绘制内容叠加到摄像头图像上
    # 这一套流程，输入 img（实时摄像头捕获的图像）和 imgCanvas（当前绘制的内容、即画布），输出 img（合成后的图像，包含摄像头图像和绘制内容）

    # 设置顶部工具栏图像
    header = cv2.resize(header, (1280, 125))  # 调整工具栏大小
    img[0:125, 0:1280] = header  # 将工具栏放置在顶部

    cv2.imshow("Image", img)  # 显示合成后的图像窗口，包含摄像头图像和绘制内容
    cv2.imshow("Canvas", imgCanvas)  # 显示当前的绘制画布，仅包含绘制的内容
    cv2.imshow("Inv", imgInv)  # 显示反转后的二值图像，用于调试和查看掩膜效果
    cv2.waitKey(1)
