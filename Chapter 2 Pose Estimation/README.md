# mp.solutions.pose

`mpPose = mp.solutions.pose` 导入 MediaPipe 的姿势估计模块

`pose = mpPose.Pose()` 创建一个姿势检测对象，用于处理图像并检测人体姿势

`results = pose.process(imgRGB)` 处理图像，检测姿势