# Tracking-RT 项目管理

远程仓库：[I-sLanD/Tracking-RT](https://github.com/I-sLanD/Tracking-RT)。
本地项目目录沿用 `jetson-perception-cpp`，无需改名或搬动已有模型、工具链。

## 哪些文件进入仓库

| 内容 | 管理方式 |
|---|---|
| `src/`、`tests/`、`tools/` | 提交源代码、算法测试、构建与验证工具 |
| `CMakeLists.txt`、环境文件、`.vscode/` | 提交构建和开发配置；新机器调整 Conda 路径 |
| `docs/`、`reports/` | 提交设计、验证结论、环境与性能记录 |
| `models/` 下的 JSON、类别文件 | 提交来源、预处理、哈希与导出参数 |
| 权重、ONNX、TensorRT engine | 保留在本机，通过下载和导出工具恢复 |
| `third_party/`、`build/`、`output/`、视频与帧目录 | 由忽略规则排除 |

现有样例图片随源码保留，来源与哈希记录在 `models/sources.json`。
第三方来源与许可记录见 `THIRD_PARTY.md`。添加新依赖时同步补全记录。

## 分支与提交

`main` 保存可验证的版本。每项改动开一个分支，完成测试后通过 Pull Request 合入。
建议把 nndeploy 接入、时间间隔预测、ReID 批处理、Jetson 后端分别拆成独立任务。

在 VS Code 中打开项目，左侧“源代码管理”可以查看文件差异、暂存、提交与同步。
终端使用同一套 Git 仓库；下面命令兼容当前安装的 Git：

```powershell
git checkout main
git pull --ff-only
git checkout -b feat/nndeploy-pipeline

# 完成一个可独立验证的改动后，先检查实际差异。
git status --short
git diff
git add src tests CMakeLists.txt docs
git diff --cached
git commit -m "refactor: integrate the nndeploy pipeline"
git push -u origin feat/nndeploy-pipeline
```

提交消息按目的写，例如 `feat: support elapsed-time prediction`、
`fix: advance tracker on empty detections`、`perf: batch ReID inference`。
一次提交尽量对应一个完整改动，验证报告记录相同输入与参数。

## 自动检查与本机验证

`.github/workflows/tracker-tests.yml` 在 `main` 的推送、面向 `main` 的 PR 和手动触发时
编译并运行现有 DeepSORT 算法测试。该任务只需要 Eigen，不下载模型，也不使用 GPU。
Linux 自动检查的执行结果以 GitHub Actions 页面为准。

本机完整感知链路仍使用 Anaconda 环境和已有验证入口：

```powershell
conda activate jetson-perception
powershell -ExecutionPolicy Bypass -File tools/build.ps1
powershell -ExecutionPolicy Bypass -File tools/validate_tracking.ps1
```

自动算法测试不能代替 Windows DirectML 或 Jetson TensorRT 的设备验证。
涉及推理或性能变化时，在 PR 中记录硬件、模型哈希、视频、后端、指标和计时边界。

## 远程登录

`origin` 使用 HTTPS 地址 `https://github.com/I-sLanD/Tracking-RT.git`。
Git 提交作者配置与远程登录是两个设置；配置作者邮箱不会获得仓库写入权限。
推送需要登录 `I-sLanD`，或使用已获得此仓库写入权限的账号。
登录在 Git Credential Manager、VS Code 或 GitHub CLI 的授权页面完成，凭据不写入源码。

官方说明：[上传已有本地项目](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、
[管理远程仓库](https://docs.github.com/en/get-started/git-basics/managing-remote-repositories)。
