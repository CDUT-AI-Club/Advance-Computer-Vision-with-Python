import cv2
import mediapipe as mp
import time

# 打开视频文件
cap = cv2.VideoCapture(
    "E:\\Advance Computer Vision with Python\\main\\Chapter 3 Face Detection\\Videos\\4.mp4"
)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

pTime = 0

# 初始化 MediaPipe 的绘图工具和面部网格模型
mpDraw = mp.solutions.drawing_utils  # 导入 MediaPipe 的绘图工具模块
mpFaceMesh = mp.solutions.face_mesh # 导入 MediaPipe 的面部网格模块，用于检测和处理面部特征点
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) # 初始化面部网格模型，设置最多检测两张人脸
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2) # 创建一个绘图规格对象，用于定义标志点和连接线的绘制样式
# thickness 指定线的厚度，circle_radius 指定标志点的半径

while True:
    print("Reading video frame...")
    success, img = cap.read()
    print("Read success:", success)
    if not success:
        print("Finished processing video or error occurred.")
        break

    # 将图像转换为 RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 处理图像以检测面部网格
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # 绘制面部网格
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            # mpDraw.draw_landmarks 调用 MediaPipe 的绘图函数，用于在图像上绘制标志点和连接
            # img: 要在其上绘制标志点的图像
            # faceLms: 检测到的人脸标志点集合
            # mpFaceMesh.FACEMESH_TESSELATION: 指定要绘制的连接类型，这里是面部网格的细分连接
            # drawSpec: 定义标志点和连接线的绘制样式（如厚度和圆圈半径）
            # 后面两个drawSpec，第一个用于定义标志点（关键点）的绘制样式，比如圆圈的半径和颜色，第二个用于定义连接线的绘制样式，比如线的厚度和颜色
            # 可以选择分别定义：
            # drawSpecPoints = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2) # 绿色
            # drawSpecLines = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1) # 蓝色
            # 其中对于drawSpecLines 中定义 circle_radius=2 是没有效果的，circle_radius 只影响标志点（关键点）的绘制，而不会影响连接线的绘制，因此，在 drawSpecLines 中设置 circle_radius 没有意义

            for id, lm in enumerate(faceLms.landmark):
                # 获取图像的尺寸
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                # 打印每个标志点的 ID 和坐标
                print(id, x, y)

    # 计算并显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口

    # 显示图像
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
