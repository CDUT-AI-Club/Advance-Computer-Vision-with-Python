import cv2
import mediapipe as mp
import time
import math


# 姿势检测类
class poseDetector:
    # 以下为老版本代码，部分参数已经弃用
    # def __init__(
    #     self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5
    # ):
    #     # 初始化参数
    #     self.mode = mode  # 静态图像模式
    #     self.upBody = upBody  # 是否只检测上半身
    #     self.smooth = smooth  # 平滑处理
    #     self.detectionCon = detectionCon  # 检测置信度
    #     self.trackCon = trackCon  # 跟踪置信度
    #     self.mpDraw = mp.solutions.drawing_utils  # Mediapipe 绘图工具
    #     self.mpPose = mp.solutions.pose  # Mediapipe 姿势检测模块
    #     self.pose = self.mpPose.Pose(
    #         self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon
    #     )

    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # 初始化姿势检测器的参数
        self.static_image_mode = static_image_mode  # 是否处理静态图像，False 表示处理视频流
        self.model_complexity = model_complexity  # 模型复杂度，0、1 或 2，越高越准确但更慢
        self.enable_segmentation = enable_segmentation  # 是否启用分割功能
        self.min_detection_confidence = min_detection_confidence  # 姿势检测的最小置信度
        self.min_tracking_confidence = min_tracking_confidence  # 姿势跟踪的最小置信度

        # 设置 MediaPipe 工具
        self.mpDraw = mp.solutions.drawing_utils  # MediaPipe 绘图工具
        self.mpPose = mp.solutions.pose  # MediaPipe 姿势估计模块

        # 创建姿势估计对象
        self.pose = self.mpPose.Pose(
            static_image_mode=self.static_image_mode,  # 静态图像模式设置
            model_complexity=self.model_complexity,  # 模型复杂度设置
            enable_segmentation=self.enable_segmentation,  # 分割功能设置
            min_detection_confidence=self.min_detection_confidence,  # 检测置信度阈值
            min_tracking_confidence=self.min_tracking_confidence  # 跟踪置信度阈值
        )


    def findPose(self, img, draw=True):
        # 将图像从 BGR 转换为 RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 处理图像，检测姿势
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                # 绘制姿势连接
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )
        return img

    def findPosition(self, img, draw=True):
        # 初始化列表存储关键点位置
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # 计算关键点在图像中的位置
                self.lmList.append([id, cx, cy])
                if draw:
                    # 在图像上绘制关键点
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        # 获取关键点坐标
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        # x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        # x3, y3 = self.lmList[p3][1], self.lmList[p3][2]

        # 计算角度
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        # 两个向量 (x1-x2, y1-y2), (x3-x2, y3-y2)
        # 使用 atan2 计算每个向量与 x 轴的角度
        if angle < 0:
            angle += 360

        # 绘制
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(
                img,
                str(int(angle)),
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
        return angle


# 主函数
def main():
    cap = cv2.VideoCapture(
        "E:\\Advance Computer Vision with Python\\Chapter 2 Pose Estimation\\PoseVideos\\5.mp4"
    )  # 打开视频文件
    pTime = 0  # 前一帧时间
    detector = poseDetector()  # 创建姿势检测器
    while True:
        success, img = cap.read()  # 读取视频帧
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        img = detector.findPose(img)  # 检测姿势
        if not success:
            print("Failed to read frame")
            break
        lmList = detector.findPosition(img, draw=False)  # 获取关键点位置
        if len(lmList) != 0:
            print(lmList[14])  # 打印关键点信息
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # 计算帧率
        pTime = cTime
        cv2.putText(
            img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
        )
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
        cv2.imshow("Image", img)  # 显示图像
        cv2.waitKey(1)  # 等待按键以显示下一帧


if __name__ == "__main__":
    main()
