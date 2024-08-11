import cv2
import mediapipe as mp
import time

# 打开视频文件
cap = cv2.VideoCapture(
    "E:\\Advance Computer Vision with Python\\main\\Chapter 3 Face Detection\\Videos\\4.mp4"
)

pTime = 0  # 上一帧的时间

# 初始化MediaPipe的人脸检测模块
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)  # 创建一个 MediaPipe 的人脸检测器对象并设置检测置信度阈值为0.75

while True:
    success, img = cap.read()  # 读取视频帧
    if not success:
        print("Failed to read frame")
        break

    # 将图像转换为RGB格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)  # 处理图像，进行人脸检测

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection) #使用 MediaPipe 提供的工具在图像上绘制检测到的人脸的边界框和关键点
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

            # 获取人脸检测的边界框信息
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # 绘制边界框
            cv2.rectangle(img, bbox, (255, 0, 255), 2)

            # 显示检测置信度
            cv2.putText(
                img,
                f"{int(detection.score[0] * 100)}%",
                (bbox[0], bbox[1] - 20),
                cv2.FONT_HERSHEY_PLAIN,
                5,
                (255, 0, 255),
                5,
            )
            # 第一个5为字体大小，第二个5为字体粗细

    # 计算并显示帧率FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5
    )

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口

    # 显示图像
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
