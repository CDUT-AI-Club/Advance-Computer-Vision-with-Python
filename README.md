# Advance Computer Vision with Python

This repository is suitable for beginners in computer vision. The code is modified based on [Advance Computer Vision with Python](https://www.computervision.zone/courses/advance-computer-vision-with-python/).

This repository is intended for use in the 2024 Chengdu University of Technology AI Association technical training.

This is the English version of the repository. You can find the Chinese version here: [Chinese version](https://github.com/Diraw/Advance-Computer-Vision-with-Python/tree/main)

这是本仓库的英文版本，你可以在这里找到中文版本：[中文版本](https://github.com/Diraw/Advance-Computer-Vision-with-Python/tree/main)

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

## Conda Environment Setup

```bash
conda create -n visionpy python=3.8
conda activate visionpy
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python==4.10.0
pip install mediapipe==0.10.10
```