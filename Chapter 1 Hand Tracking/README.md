# mp.solutions.hands

`mpHands = mp.solutions.hands` 引用 MediaPipe 的手部解决方案模块

`hands = mpHands.Hands()` 创建一个 Hands 对象，用于检测和跟踪手部关键点

`results = hands.process(imgRGB)` 处理图像以检测手部

`for handLms in results.multi_hand_landmarks` results.multi_hand_landmarks会返回一个列表，其中包含检测到的每只手的关键点信息

`for id, lm in enumerate(handLms.landmark)` id 是手部关键点的索引，lm 是 landmark 的缩写，表示手部关键点的坐标信息

lm.x 和 lm.y 是关键点的归一化坐标，范围在 0 到 1 之间。通过乘以图像的宽度和高度，可以将它们转换为图像中的像素坐标

`mpHands.Hands()` 对象的参数：

```python
static_image_mode: 是否将每帧作为静态图像处理
max_num_hands: 最大检测手数
min_detection_confidence: 检测置信度阈值
min_tracking_confidence: 跟踪置信度阈值
```

# mp.solutions.drawing_utils

`mpDraw = mp.solutions.drawing_utils` 引用绘图工具，用于在图像上绘制检测到的手部关键点和连接线

`mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)` 绘制手部关键点和连接线，img 要绘制的图像，handLms 手部关键点的坐标，mpHands.HAND_CONNECTIONS 定义手部关键点之间的连接关系，用于绘制骨架结构

# pycache文件夹

当导入一个 .py 文件时，Python 会编译它，并将编译后的字节码存储在 __pycache__ 目录中。这有助于提高程序的执行效率，因为下次运行时可以直接使用编译后的字节码，而不必重新编译源代码

“Cache” 是指缓存，一种用于临时存储数据的机制，以便更快速地访问。缓存可以减少数据的重复计算或从慢速存储设备读取的次数，从而提高程序性能

对于HandTrackingModule.py而言，当你在其他文件中导入这个模块时，只有类 handDetector 和其中的方法会被使用，而 main() 函数不会，main() 函数只有在直接运行该脚本时才会执行