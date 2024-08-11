import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(
        self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5
    ):
        # 初始化参数
        self.staticMode = staticMode  # 是否使用静态模式
        self.maxFaces = maxFaces  # 最大检测人脸数量
        self.minDetectionCon = minDetectionCon  # 最小检测置信度
        self.minTrackCon = minTrackCon  # 最小跟踪置信度

        # 初始化 MediaPipe 的绘图工具和面部网格模型
        self.mpDraw = mp.solutions.drawing_utils  # 绘图工具
        self.mpFaceMesh = mp.solutions.face_mesh  # 面部网格模块
        # self.faceMesh = self.mpFaceMesh.FaceMesh(
        #     self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon
        # )
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon,
        )

        self.drawSpec = self.mpDraw.DrawingSpec(
            thickness=1, circle_radius=2
        )  # 绘图规格

    def findFaceMesh(self, img, draw=True):
        # 将图像转换为 RGB
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 处理图像以检测面部网格
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []  # 存储检测到的面部特征点
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    # 绘制面部网格
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.drawSpec,
                        self.drawSpec,
                    )
                face = []  # 存储单个人脸的特征点
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape  # 获取图像的高、宽、通道数
                    x, y = int(lm.x * iw), int(lm.y * ih)  # 将归一化坐标转换为像素坐标
                    face.append([x, y])  # 添加特征点坐标
                faces.append(face)  # 添加到面部列表
        return img, faces  # 返回图像和面部特征点


def main():
    # 打开视频文件
    cap = cv2.VideoCapture(
        "E:\\Advance Computer Vision with Python\\main\\Chapter 3 Face Detection\\Videos\\4.mp4"
    )
    pTime = 0  # 上一帧的时间
    detector = FaceMeshDetector(maxFaces=2)  # 初始化面部网格检测器
    while True:
        success, img = cap.read()
        if not success:
            break
        img, faces = detector.findFaceMesh(img)  # 检测面部网格
        if len(faces) != 0:
            print(faces[0])  # 打印第一个面部的特征点
        cTime = time.time()  # 当前时间
        fps = 1 / (cTime - pTime)  # 计算帧率
        pTime = cTime  # 更新上一帧的时间
        cv2.putText(
            img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
        )  # 在图像上显示帧率
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
        cv2.imshow("Image", img)  # 显示图像
        cv2.waitKey(1)  # 等待键盘输入


if __name__ == "__main__":
    main()
