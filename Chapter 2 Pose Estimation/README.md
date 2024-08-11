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

如果出现如下报错：

> cv2.error: OpenCV(4.10.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1301: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'

请把终端移动到最外层文件夹运行（对于vscode就是打开最外层文件夹），我也不是很清楚为什么会出现这种bug

以及视频文件的路径请使用**绝对路径**