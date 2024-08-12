[中文版本](https://github.com/Diraw/Advance-Computer-Vision-with-Python/tree/main) | English version

# Advance Computer Vision with Python

This project offers a simple and straightforward introductory tutorial for beginners in computer vision. Through practical code exercises and comprehensive annotations, users can gradually master core technologies such as hand gesture recognition, pose estimation, and face detection.

The code is modified and optimized based on [Advance Computer Vision with Python](https://www.computervision.zone/courses/advance-computer-vision-with-python/) to lower the barriers to learning and implementation.

This repository is intended for use in the 2024 Chengdu University of Technology AI Club technical training.

<table width="100%">
  <tr>
    <td width="60%"><img src="./pics/手势控制电脑音量.gif" alt="手势控制电脑音量" style="width: 100%;"></td>
    <td width="40%"><img src="./pics/finger_counter.gif" alt="finger_counter" style="width: 100%;"></td>
  </tr>
  <tr>
    <td colspan="2"><img src="./pics/AI Virtual Mouse.gif" alt="AI Virtual Mouse" style="width: 100%;"></td>
  </tr>
  <tr>
    <td colspan="2"><img src="./pics/AI Virtual Painter.gif" alt="AI Virtual Painter" style="width: 100%;"></td>
  </tr>
</table>


## Advantages of This Repository

1. Added extensive comments to the source code, explaining various issues beginners might encounter. For example:

![1](./pics/1.png)

![2](./pics/2.png)

2. Included more user-friendly designs based on the original code, such as allowing manual window resizing and setting exit keys.

3. All code runs successfully. Any changes between old and new code are noted in the `README.md` file of the respective folder. For example:

![3](./Chapter%204%20Face%20Mesh/pics/关键字传参.png)

4. Each chapter includes a `README.md` file summarizing important concepts for easy review.

## Project Dependencies

- Python: `3.8`
- OpenCV: `4.10.0`
- MediaPipe: `0.10.10`
- Pycaw: `20240210`

## Conda Environment Setup

```bash
conda create -n visionpy python=3.8
conda activate visionpy
pip install opencv-python==4.10.0
pip install mediapipe==0.10.10
pip install pycaw==20240210
```