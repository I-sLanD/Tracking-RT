# Tracking-RT：YOLO26 + DeepSORT C++ 感知基线

仓库：[I-sLanD/Tracking-RT](https://github.com/I-sLanD/Tracking-RT)。
Git 分支、提交、PR 和验证流程见 [项目管理说明](docs/github-workflow.md)。

先在当前 Windows / RTX 4080 上完成原生 C++ 检测、行人 ReID 和多目标跟踪，作为后续感知算法改进与 Jetson 端侧优化的基线。

当前实现：图片解码 → RGB Letterbox → FP32 NCHW → ONNX Runtime → 检测框解析与原图坐标还原。提供 CPU 与 Windows DirectML GPU 后端，输出检测图、CSV、张量和分阶段计时。

新增连续帧链路：YOLO26 检测行人 → OSNet x0.25 外观特征 → DeepSORT 预测/关联/轨迹管理。输出轨迹 ID、MOT 格式结果、逐帧日志、各阶段耗时以及带 ID 和轨迹线的演示视频。

## 开发环境

- Python：Anaconda 管理的 `jetson-perception` 环境，Python 3.11。
- Python 参考推理：CPU PyTorch 2.4.1 / torchvision 0.19.1。
- IDE：VS Code，已配置解释器、C++ IntelliSense、构建任务和 LLDB-DAP 调试入口。
- C++：项目内 LLVM-MinGW 20260922，C++17，CMake / Ninja。
- 原生推理：ONNX Runtime 1.24.4。当前 Windows 系统自带 DirectML 1.15.5。

Python 负责模型导出、独立结果对照和离线视频解码/渲染；检测、ReID、跟踪全部在 C++ 可执行程序内运行，直接调用原生 ONNX Runtime。

## DeepSORT 使用

本机依赖、模型和示例视频已准备好。在 VS Code 的“运行任务”中选择 `DeepSORT: C++ GPU (DirectML)` 即可构建并处理示例；选择 `DeepSORT: full validation` 可运行完整对照。F5 中新增 `Debug DeepSORT C++ (CPU)`。

终端入口：

```powershell
conda activate jetson-perception
powershell -ExecutionPolicy Bypass -File tools/build.ps1
powershell -ExecutionPolicy Bypass -File tools/track.ps1 -Provider dml
powershell -ExecutionPolicy Bypass -File tools/validate_tracking.ps1
```

换成自己的视频时，给它一个新的帧目录和输出名称：

```powershell
powershell -ExecutionPolicy Bypass -File tools/track.ps1 -Provider dml -Video "D:\videos\people.mp4" -Frames "assets\my-video-frames" -Limit 300 -OutputName my-video
```

不要把不同视频放入同一个帧目录。离线解码不跳帧，文件使用补零编号；传入源视频 FPS 记录时间戳。当前假设固定帧率，尚未支持变帧率/丢帧时的真实时间间隔预测。

`output/tracking-dml` 包含：

| 文件 | 用途 |
|---|---|
| `tracked.mp4` / `preview.png` | 已确认且本帧观测到的行人 ID、轨迹线 |
| `tracks.jsonl` | 帧号、时间戳、检测观测、ID、轨迹状态、匹配检测索引、预测框 |
| `mot.txt` | MOT 格式的已确认且本帧观测到的轨迹 |
| `timings.csv` / `summary.json` | 检测、ReID、关联、PNG 解码的独立计时 |
| `features.f32` | `-DumpFeatures` 时保存的 ReID 特征，供固定输入对照 |

`created_tracks` 是分配过的轨迹 ID 数，包含临时轨迹和重复建轨，不能当作实际人数。未更新轨迹继续保留预测状态；`observed=false` 时不画框、不写入 MOT 观测。原版线性 Kalman 模型可能给长期丢失轨迹预测出非正尺寸，内部状态保留用于原版对照，`valid_box=false` 的框不参与展示或 MOT 输出。

## 跟踪模型和参数

外观模型来自 [Torchreid 作者模型库](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)：MSMT17 `combineall=True` 训练的 `osnet_x0_25`。原始权重与导出的 ONNX 在 `models`，来源、哈希和预处理规则见 `tracking-sources.json` / `reid-export.json`。

ReID 输入 `[1,3,256,128]`：裁剪原图行人框，RGB 双线性缩放、ImageNet 均值/标准差；输出 `[1,512]` 后进行 L2 归一化。采用固定 batch=1，逐人调用，便于建立可对照的初始版本。此模型只适用于行人外观，当前过滤 COCO `person` 类别。

DeepSORT 保留 8 维 XYAH 恒速度 Kalman、4 维 Mahalanobis 门控、图库最小余弦距离、按未更新时间匹配的级联、匈牙利分配、IoU 回退、Tentative/Confirmed/Deleted 状态。算法参考 [DeepSORT 论文](https://arxiv.org/abs/1703.07402)，并与[作者实现](https://github.com/nwojke/deep_sort)独立对照。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `-Threshold` | 0.25 | 行人检测置信度下限 |
| `-MaxCosine` | 0.2 | 允许的最大外观距离 |
| `-MaxIou` | 0.7 | IoU 回退的最大 `1-IoU` |
| `-NInit` | 3 | 连续观测确认轨迹；设置 1 时本项目立即确认 |
| `-MaxAge` | 30 | 删除前最多保留的漏检帧数 |
| `-NNBudget` | 100 | 每条轨迹最多保存的特征数 |

参数通过 `tools/track.ps1` 传入 C++，每次运行记入 `summary.json`。固定 `dt=1` 为一帧，30 帧在 10 FPS 视频中代表 3 秒。参数初值未针对示例调优。

## DeepSORT 验证

完整结果见 `reports/tracking-validation.md` / `.json`。C++ 测试覆盖最优分配、外观关联、漏检恢复、门控、删除、图库上限和长时间数值稳定性。受控序列逐帧对照作者实现的 ID、状态、Kalman mean/covariance；公开视频允许新建轨迹的编号排序不同，只在出生帧固定编号对应关系，之后检查整条轨迹的关联与生命周期，禁止重映射。另对照 PyTorch 与 C++ ReID 输出；CPU/GPU 分别执行视频，检查检测、特征和关联一致性，ORT profile 确认两条模型链路实际使用 GPU。

演示采用 [OpenCV 原始视频样例 vtest.avi](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi) 前 180 帧。它没有真实 ID 标注，本次只验证实现行为，不据此给出 IDF1 / HOTA / MOTA。计时来自未开启 profile 的运行，感知流水线排除初始化、暖机、图片解码、日志、视频 IO 和渲染，不能换算为摄像头端到端 FPS。

新增依赖的复现入口：

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_tracking.ps1
```

依赖沿用现有 Anaconda 环境；增加 Eigen 头文件、作者参考源码以及模型。许可与来源见 `THIRD_PARTY.md`。

## 参考模型

官方文件：[yolo26n.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)，保存于 `models/yolo26n.pt`。

官方 SHA-256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

显式导出 NMS-free 分支：`nms=False`、固定 batch=1、640×640、opset=17、FP32。输入 `[1,3,640,640]`，输出 `[1,300,6]`，每行为 `[x1,y1,x2,y2,confidence,class_id]`。程序检查实际输出形状，并且不重复执行 NMS。

配置、版本和 ONNX 哈希记录在 `models/export.json`；下载来源与哈希记录在 `models/sources.json`。导出规则参考 [Ultralytics 官方文档](https://docs.ultralytics.com/modes/export)。

## 在 VS Code 中运行

打开此目录。`Ctrl+Shift+B` 编译 C++；“运行任务”中可选：

- `YOLO26: export official reference`：重新导出模型。
- `YOLO26: C++ CPU`：原生 CPU 推理。
- `YOLO26: C++ GPU (DirectML)`：当前设备 GPU 推理。
- `YOLO26: full validation`：构建、两张图片的 CPU/GPU 测试、Python 对照与报告。

F5 使用 `Debug YOLO26 C++ (CPU)` 调试配置。当前机器解释器路径已经写入 `.vscode/settings.json`；迁移机器时需要更换解释器和 Conda 路径。

## 从终端运行

在此项目目录的 PowerShell 中执行：

```powershell
conda activate jetson-perception
python tools/export_model.py
powershell -ExecutionPolicy Bypass -File tools/build.ps1
powershell -ExecutionPolicy Bypass -File tools/run.ps1 -Provider cpu
powershell -ExecutionPolicy Bypass -File tools/run.ps1 -Provider dml
powershell -ExecutionPolicy Bypass -File tools/validate.ps1
```

换图片：

```powershell
powershell -ExecutionPolicy Bypass -File tools/run.ps1 -Provider dml -Image "你的图片路径.jpg" -OutputName custom
```

`output/cpu` 和 `output/dml` 中的 `annotated.png` 为检测结果图，绿色为行人、橙色为其他类别；标签、分数和坐标见 `detections.csv` / `result.json`。

## 复现安装

```powershell
conda create -n jetson-perception python=3.11 pip -y --override-channels -c conda-forge
conda activate jetson-perception
python -m pip install "numpy<2" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python tools/download_dependencies.py
python tools/export_model.py
```

`environment.yml` 管理 Python 版本；完成安装后会保存 `requirements-lock.txt` 和 Conda 明细用于记录本次环境。

本机 GPU 验证使用 Windows DirectML。其他 Windows 设备需要兼容的 DirectX 12 / DirectML 环境；详见 [ONNX Runtime DirectML 要求](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)。DeepSORT 算法测试已在 GitHub Actions 的 Ubuntu 24.04 环境通过。完整 YOLO26 / OSNet 推理链路及 Jetson 后端尚未在 Linux 上验证；Linux CMake 路径预留了同一 C API 的 `.so` 加载方式。

## 验证与计时边界

结果在 `reports/validation.md` 和 `reports/validation.json`。验证使用官方 `bus.jpg` 与 `zidane.jpg`，覆盖横向和纵向 Letterbox 填充，检查预处理、模型输出、检测框解析、CPU/GPU 一致性及 GPU provider 实际执行。

计时包括内存中图片预处理、同步推理、检测解析；排除模型初始化、读文件、图片解码、画框与保存。重复同一张图片的计时用于初步对照，不等同于摄像头视频帧率或 Jetson 性能。

## 下一阶段接口

保持 YOLO26 为固定检测器，利用 `tracks.jsonl` 和 `features.f32` 保存检测与外观观测，隔离跟踪关联的变化。核心在 `src/deepsort.*`，原生模型及预处理在 `src/inference.hpp`，流水线在 `src/track_main.cpp`。

后续顺序：原版 DeepSORT 基线 → 实际时间间隔的 Kalman 预测 → ReID 特征质量管理 → 按跟踪状态调度 ReID → GPU 裁剪和预处理 → Jetson TensorRT 验证。每项保留独立开关与对照结果。

当前已完成 DeepSORT / ReID 基线。摄像头、云台、TensorRT 和 Jetson 端侧性能尚未验证。第三方文件与权重保留原有许可证；原生依赖的许可证位于各自包内。
