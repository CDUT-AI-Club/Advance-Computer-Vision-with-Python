import cv2
import mediapipe as mp
import time
import math


# 姿势检测类
class poseDetector:
    def __init__(
        self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5
    ):
        # 初始化参数
        self.mode = mode  # 静态图像模式
        self.upBody = upBody  # 是否只检测上半身
        self.smooth = smooth  # 平滑处理
        self.detectionCon = detectionCon  # 检测置信度
        self.trackCon = trackCon  # 跟踪置信度
        self.mpDraw = mp.solutions.drawing_utils  # Mediapipe 绘图工具
        self.mpPose = mp.solutions.pose  # Mediapipe 姿势检测模块
        self.pose = self.mpPose.Pose(
            self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon
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
        # 计算角度
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
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
    cap = cv2.VideoCapture("PoseVideos/1.mp4")  # 打开视频文件
    pTime = 0  # 前一帧时间
    detector = poseDetector()  # 创建姿势检测器
    while True:
        success, img = cap.read()  # 读取视频帧
        img = detector.findPose(img)  # 检测姿势
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
        cv2.imshow("Image", img)  # 显示图像
        cv2.waitKey(1)  # 等待按键以显示下一帧


if __name__ == "__main__":
    main()
