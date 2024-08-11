import cv2
import mediapipe as mp
import time


# 定义一个人脸检测类
class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        # 初始化检测置信度
        self.minDetectionCon = minDetectionCon

        # 初始化MediaPipe的人脸检测模块
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        # 将图像转换为RGB格式
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)  # 进行人脸检测
        bboxs = []  # 存储检测到的边界框

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # 获取人脸检测的边界框信息
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                bboxs.append([id, bbox, detection.score])  # 添加到边界框列表

                if draw:
                    img = self.fancyDraw(img, bbox)  # 绘制边界框
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
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        # 这个 fancyDraw 函数通过在矩形框的四个角上绘制短线来实现自定义样式，与普通的矩形框相比，增加了视觉上的变化
        # 1、矩形框：使用 cv2.rectangle 绘制标准矩形框
        # 2、角线：在矩形的四个角上绘制短线段，使边框看起来更有设计感
        # l 角线的长度，t 角线的粗细，rt 矩形框的粗细（length 长度，thickness 厚度）

        # 自定义绘制边框的样式
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)  # 绘制矩形框
        # 绘制四个角的线条
        # 左上角
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # 右上角
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # 左下角
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # 右下角
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        
        return img


def main():
    # 打开视频文件
    cap = cv2.VideoCapture(
        "E:\\Advance Computer Vision with Python\\main\\Chapter 3 Face Detection\\Videos\\4.mp4"
    )
    pTime = 0  # 上一帧时间
    detector = FaceDetector()  # 创建人脸检测器对象

    while True:
        success, img = cap.read()  # 读取视频帧
        img, bboxs = detector.findFaces(img)  # 检测人脸并获取边界框
        print(bboxs)  # 打印边界框信息

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
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
