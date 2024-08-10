# mp.solutions.pose

`mpPose = mp.solutions.pose` 导入 MediaPipe 的姿势估计模块

`pose = mpPose.Pose()` 创建一个姿势检测对象，用于处理图像并检测人体姿势

`results = pose.process(imgRGB)` 处理图像，检测姿势

`mpPose.Pose()` 对象的参数

```python
static_image_mode=self.static_image_mode,  # 静态图像模式设置
model_complexity=self.model_complexity,  # 模型复杂度设置
enable_segmentation=self.enable_segmentation,  # 分割功能设置
min_detection_confidence=self.min_detection_confidence,  # 检测置信度阈值
min_tracking_confidence=self.min_tracking_confidence  # 跟踪置信度阈值
```

# 注意事项

用vscode执行代码时，请让终端在`Advance Computer Vision with Python`文件夹下（即vs打开这个整体的文件夹），以及视频路径请使用**绝对路径**，否则，将可能出现一些莫名其妙的报错

比如你让终端在`Chapter 2 Pose Estimation`文件夹下运行，opencv的GUI就要报错，**日怪得很**