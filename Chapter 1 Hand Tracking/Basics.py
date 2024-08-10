# cv2.__version__ = 4.10.0, mp.__version__ = 0.10.14

import cv2
import mediapipe as mp
import time

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0是默认摄像头

# 初始化手部检测模块
mpHands = mp.solutions.hands  # 引用 MediaPipe 的手部解决方案模块
hands = mpHands.Hands()  # 创建一个 Hands 对象，用于检测和跟踪手部关键点
mpDraw = mp.solutions.drawing_utils  # 引用绘图工具，用于在图像上绘制检测到的手部关键点和连接线

# 初始化时间变量用于计算帧率
pTime = 0  # 表示前一帧的时间
cTime = 0  # 表示当前帧的时间
# cTime - pTime 计算时间差，从而计算帧率。最后将 cTime 赋值给 pTime，以便在下一次循环时使用

while True:
    # 读取摄像头图像
    success, img = cap.read()
    # success：一个布尔值，表示是否成功读取帧
    # img：读取的图像帧，如果读取失败，这个值可能为空

    # 水平翻转图像
    img = cv2.flip(img, 1)
    # 第 0 维表示垂直方向（高度），对应图像的行数，上下
    # 第 1 维表示水平方向（宽度），对应图像的列数，左右

    # 将图像从 BGR 格式转换为 RGB 格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # BGR 是图像在 OpenCV 中的默认颜色格式，代表蓝色（Blue）、绿色（Green）、红色（Red）。这种格式与通常使用的 RGB（红、绿、蓝）顺序相反。转换为 RGB 是因为许多图像处理库（如 MediaPipe）使用这种格式进行处理

    # 处理图像以检测手部
    results = hands.process(imgRGB)

    # 如果检测到手部
    if results.multi_hand_landmarks:
        # 遍历检测到的每只手
        for handLms in results.multi_hand_landmarks:
            # results.multi_hand_landmarks 会返回一个列表，其中包含检测到的每只手的关键点信息。如果检测到多只手，它会包含多个元素，每个元素代表一只手的所有关键点

            # 遍历手部关键点
            for id, lm in enumerate(handLms.landmark):  
                # enumerate 返回一个迭代器，每次迭代返回一个包含索引和值的元组
                # id 是手部关键点的索引，lm 是 landmark 的缩写，表示手部关键点的坐标信息

                # 获取图像的尺寸
                h, w, c = (img.shape) 
                # img.shape 返回一个包含图像维度的元组，具体包括：高度（行数）、宽度（列数）、通道数（如 RGB 图像的通道数为 3）

                # 计算关键点在图像中的坐标
                cx, cy = int(lm.x * w), int(lm.y * h)
                # lm.x 和 lm.y 是关键点的归一化坐标，范围在 0 到 1 之间。通过乘以图像的宽度和高度，可以将它们转换为图像中的像素坐标

                print(id, cx, cy)

                # 在关键点处画一个圆圈
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), -1)
                # img 表示要绘制图像的地方，(cx, cy) 圆心的坐标，15 圆的半径
                # (255, 0, 255) 圆的颜色（BGR格式），这里是紫色
                # 红色：(0, 0, 255)、绿色：(0, 255, 0)、蓝色：(255, 0, 0)、黄色：(0, 255, 255)、青色：(255, 255, 0)、品红：(255, 0, 255)、白色：(255, 255, 255)、黑色：(0, 0, 0)
                # cv2.FILLED 或 -1 填充圆的实心样式，也可以为具体的数字（值为边框厚度）

            # 绘制手部关键点和连接线
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # img 要绘制的图像，handLms 手部关键点的坐标
            # mpHands.HAND_CONNECTIONS 定义手部关键点之间的连接关系，用于绘制骨架结构

    # 计算帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # 在图像上显示帧率
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # img 要绘制文本的地方，str(int(fps)) 要显示的文本内容，这里是帧率的整数部分
    # (10, 70) 文本的左下角坐标，cv2.FONT_HERSHEY_PLAIN 字体样式
    # 3 字体大小，(255, 0, 255) 文本颜色（紫色，BGR格式），3 文本的粗细
    # cv2.putText 不支持关键字传参，必须按照顺序提供参数

    # 显示图像
    cv2.imshow("Image", img)  # 在窗口中显示图像，窗口标题为“Image”
    cv2.waitKey(1) # 等待键盘事件，参数为 1 表示等待 1 毫秒
    # 它也允许图像窗口响应用户输入（如关闭窗口）

    # 检测退出键
    if cv2.waitKey(1) & 0xFF == ord("q"):  # ord('q') 获取字符 'q' 的 ASCII 值
        # cv2.waitKey(1) & 0xFF 用来读取键盘输入
        # cv2.waitKey(1) 返回的是一个 32 位整数，其中低 8 位是实际的键值，& 0xFF 是一个位运算，用于提取这 8 位
        # "低 8 位"指的是一个数值的二进制表示中最右边的 8 位。这些位表示数值的较小部分，与"高 8 位"（最左边的 8 位）相对，后者表示数值的较大部分。对于 32 位整数来说，低 8 位用于表示键盘输入的实际键值
        break

cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
