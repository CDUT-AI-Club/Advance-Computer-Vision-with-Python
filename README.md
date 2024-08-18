中文版本 | [English version](https://github.com/Diraw/Advance-Computer-Vision-with-Python/tree/main_en)

# Advance Computer Vision with Python

本项目为计算机视觉初学者提供了一套简单快捷的入门教程，通过实际的代码练习和丰富的注释，用户可以逐步掌握手势识别、姿态估计、面部检测等核心技术

本项目代码基于 [Advance Computer Vision with Python](https://www.computervision.zone/courses/advance-computer-vision-with-python/) 进行修改和优化，特别适配于国内用户的开发环境，旨在降低学习和实现的门槛

本项目计划用于2024成都理工大学人工智能协会技术培训使用

**在线演示网站**：[手部特征点检测](https://vision-py-quick-start.top/)，网站代码放在[GitHub仓库](https://github.com/Diraw/vision-py-quick-start)，个人web能力太烂了，欢迎大家PR QAQ

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

## 本仓库的优势

1. 在源代码的基础上加入了非常多的注释，对于初学者会遇到的各种问题都做了解释。例如：

![1](./pics/1.png)

![2](./pics/2.png)

2. 在源代码的基础上加入了更多人性化的设计，例如使窗口大小可手动调整，设置退出键等操作

3. 所有代码均可成功运行。在遇到新旧代码变动的情况，会在相应文件夹的 `README.md` 文件中指出，例如：

![3](./Chapter%204%20Face%20Mesh/pics/关键字传参.png)

4. 每个章节均有 `README.md` 文件，对重要的知识点进行汇总，方便复习

## 项目依赖

- Python: `3.8`
- OpenCV: `4.10.0`
- MediaPipe: `0.10.10`
- Pycaw: `20240210`
- autopy: `4.0.0`

## Conda创建环境流程

```bash
conda create -n visionpy python=3.8
conda activate visionpy
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python==4.10.0
pip install mediapipe==0.10.10
pip install pycaw==20240210
pip install autopy==4.0.0
```

