# C++ YOLO26 初步验证结果

验证通过：两个官方示例图的 C++ CPU / GPU 检测，与 Python ONNX 和 PyTorch 参考结果进行交叉检查。

当前设备：Windows 11 / Intel i7-12700KF / NVIDIA RTX 4080。GPU 后端为 ONNX Runtime DirectML。

| 图片 | 后端 | 预处理均值 ms | 推理均值 ms | 链路 P95 ms | 检测数 |
|---|---|---:|---:|---:|---:|
| bus | CPU | 7.018 | 22.240 | 32.022 | 5 |
| bus | GPU / DirectML | 6.756 | 2.634 | 10.332 | 5 |
| zidane | CPU | 5.172 | 20.427 | 29.012 | 3 |
| zidane | GPU / DirectML | 5.046 | 2.659 | 8.301 | 3 |

计时范围为内存中单张图像的预处理、同步推理与结果解析，排除模型初始化、文件读取、图像解码、画框和保存；这是桌面初验，不代表视频处理帧率或 Jetson 性能。

检查包含：RGB / NCHW / Letterbox 对照、C++ 与 Python ONNX 输出、PyTorch 检测结果、原图坐标还原、CPU 与 GPU 检测一致性，以及 GPU provider 实际执行记录。

尚未覆盖：DeepSORT、ReID、实时视频、云台、TensorRT 和 Jetson 部署。
