# mp.solutions.pose

`mpPose = mp.solutions.pose` imports the MediaPipe pose estimation module.

`pose = mpPose.Pose()` creates a pose detection object for processing images and detecting human poses.

`results = pose.process(imgRGB)` processes the image to detect poses.

Parameters for the `mpPose.Pose()` object:

```python
static_image_mode=self.static_image_mode,  # Static image mode setting
model_complexity=self.model_complexity,  # Model complexity setting
enable_segmentation=self.enable_segmentation,  # Segmentation setting
min_detection_confidence=self.min_detection_confidence,  # Detection confidence threshold
min_tracking_confidence=self.min_tracking_confidence  # Tracking confidence threshold
```

# Notes

If you encounter the following error:

> cv2.error: OpenCV(4.10.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1301: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'

Please move the terminal to the outermost folder to run (for VSCode, open the outermost folder). I'm not entirely sure why this bug occurs.

Also, please use an **absolute path** for the video file.