import cv2
import mediapipe as mp
import time

# 初始化 Mediapipe 的绘图工具和姿势检测模块
mpDraw = mp.solutions.drawing_utils # 导入 MediaPipe 的绘图工具，用于在图像上绘制检测到的姿势连接和关键点
mpPose = mp.solutions.pose  # 导入 MediaPipe 的姿势估计模块
pose = mpPose.Pose()  # 创建一个姿势检测对象，用于处理图像并检测人体姿势

# 打开视频文件
cap = cv2.VideoCapture("E:\\Advance Computer Vision with Python\\main\\Chapter 2 Pose Estimation\\PoseVideos\\3.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

pTime = 0  # 前一帧的时间

# 创建可调整大小的窗口
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()  # 读取视频帧

    if not success:
        print("Failed to read frame")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 转换为 RGB
    results = pose.process(imgRGB)  # 处理图像，检测姿势

    if results.pose_landmarks:
        # 绘制姿势连接
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # 遍历每个关键点
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape  # 获取图像尺寸
            cx, cy = int(lm.x * w), int(lm.y * h)  # 计算关键点在图像中的位置
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)# 在关键点位置绘制圆圈

    cTime = time.time()
    fps = 1 / (cTime - pTime)  # 计算帧率
    pTime = cTime

    # 在图像上显示帧率
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
