# Tracking-RT Git 操作指令

本文整理项目接入 GitHub 时实际用过的命令，并给出后续开发的常用流程。
当前本机 Git 为 `2.19.1.windows.1`，因此采用 `checkout` 和 `reset HEAD` 等兼容命令。

- 远程仓库：<https://github.com/I-sLanD/Tracking-RT>
- 本地目录：`H:\刘欣阳资料\code\jetson-perception-cpp`
- 稳定分支：`main`

先在 PowerShell 中进入项目目录：

```powershell
Set-Location -LiteralPath 'H:\刘欣阳资料\code\jetson-perception-cpp'
```

## 1. 初始化与配置：本次已完成

以下初始化操作只需执行一次。已有仓库可以直接进入后面的查看和开发流程。

```powershell
git init
git symbolic-ref HEAD refs/heads/main
```

`init` 创建本地 `.git` 仓库；`symbolic-ref` 在尚无提交时设置初始分支为 `main`。
本机 Git 版本较旧，初始化时使用了这两条命令。

设置当前仓库的提交作者，使用 GitHub noreply 邮箱：

```powershell
git config --local user.name 'I-sLanD'
git config --local user.email '83260300+I-sLanD@users.noreply.github.com'
git config --local core.autocrlf false
```

`--local` 只影响本项目；文件换行由仓库内的 `.gitattributes` 管理。
作者配置负责记录“谁提交”，远程登录负责确认“谁有权上传”。

本机曾遇到旧 Git 的 OpenSSL HTTPS 连接错误，改用 Windows 的 TLS 后端后可以访问：

```powershell
git config --local http.sslBackend schannel
```

连接远程仓库：

```powershell
git remote add origin https://github.com/I-sLanD/Tracking-RT.git
git remote -v
```

`origin` 是远程地址的名称。它已配置；以后更改地址使用：

```powershell
git remote set-url origin https://github.com/I-sLanD/Tracking-RT.git
```

## 2. 查看当前状态和改动

```powershell
git status --short --branch
git diff
git diff --cached
git diff --cached --stat
git diff --cached --check
```

| 命令 | 用途 |
|---|---|
| `status --short --branch` | 查看分支、同步状态和文件状态 |
| `diff` | 查看尚未暂存的已跟踪文件修改 |
| `diff --cached` | 查看已经暂存、将进入下一次提交的修改 |
| `diff --cached --stat` | 查看暂存改动的文件和行数统计 |
| `diff --cached --check` | 检查暂存改动的空白问题 |

新建且未跟踪的文件先出现在 `status` 中，执行 `add` 后可在暂存差异中查看。
状态中的 `??` 表示未跟踪，第一列表示暂存区状态，第二列表示工作目录状态。

## 3. 暂存和提交

本次首次建库，先检查忽略规则，再暂存全部可跟踪文件：

```powershell
git add -- .
git diff --cached --stat
git diff --cached --check
git commit -m 'chore: establish YOLO26 and DeepSORT C++ baseline'
```

日常开发建议指定这次改动的文件，例如：

```powershell
git add src/deepsort.cpp tests/tracker_tests.cpp
git diff --cached
git commit -m 'fix: handle empty observations correctly'
```

上面的日常提交消息是示例，应按真实改动改写。
`add` 把当前文件内容放入暂存区；`commit` 把暂存区保存成一个本地版本。
暂存后继续编辑文件，需要再次执行 `add` 才会纳入新的内容。

## 4. 检查模型和运行产物是否被忽略

本次用下面的命令检查了实际文件：

```powershell
git check-ignore models/yolo26n.pt models/yolo26n.onnx models/osnet_x0_25_msmt17.onnx
git check-ignore assets/pedestrians.avi build/tracker_tests.exe
git check-ignore third_party/eigen-3.4.0/Eigen/Core output/tracking-dml/summary.json
git ls-files
```

`check-ignore` 输出的路径表示被忽略；`ls-files` 列出当前进入 Git 管理的文件。
模型、视频、依赖、构建结果和运行输出由 `.gitignore` 排除，JSON 来源和验证报告保留。
忽略规则不会自动取消对已跟踪文件的管理；如曾误加文件，可保留本地文件并取消跟踪：

```powershell
git rm --cached -- models/yolo26n.onnx
```

该命令仅适用于文件已经被跟踪的情况，本项目当前无需执行。

## 5. 下载远程更新和建立分支跟踪

```powershell
git fetch origin main
git branch --set-upstream-to=origin/main main
git status --short --branch
```

`fetch` 下载远程提交并更新本地的 `origin/main`，保留当前工作文件。
`--set-upstream-to` 建立本地 `main` 与 `origin/main` 的跟踪关系。

日常同步稳定分支：

```powershell
git checkout main
git pull --ff-only
```

`pull --ff-only` 在能直接前进时更新分支；如果本地和远程各有新提交，它会停止，
这时先查看提交历史，再决定如何合并。

## 6. 开分支开发并上传

以未来的 nndeploy 接入为例：

```powershell
git checkout main
git pull --ff-only
git checkout -b feat/nndeploy-pipeline

# 修改并完成相应验证后，选择实际修改的文件。
git status --short
git add src tests CMakeLists.txt docs
git diff --cached
git commit -m 'refactor: integrate the nndeploy pipeline'
git push -u origin feat/nndeploy-pipeline
```

`checkout -b` 创建并切换到新分支；`push -u` 首次上传分支并设置跟踪关系。
之后同一分支新增提交可以执行 `git push`。在 GitHub 上创建 Pull Request，
附上验证结果，检查通过后合入 `main`。

**本次实际上传方式：** 使用已连接的 `I-sLanD` GitHub 账号通过连接工具上传，
随后用 Git 下载远程提交并对齐本地记录。此前终端中的 `git push` 因缺少本机登录而失败。
Codex 的 GitHub 账号连接与本机 Git 凭据分别管理；直接在终端推送时，需要先完成本机登录。

## 7. 查看提交历史与比较版本

```powershell
git log --oneline --decorate --graph -10
git log -1 --format='%h %s'
git show --stat HEAD
git diff main..feat/nndeploy-pipeline --stat
git diff origin/main --
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
```

`HEAD` 表示当前提交。提交 SHA 标识一个版本；`HEAD^{tree}` 标识这个版本的文件树。
相同文件树可以对应不同提交，因为父提交、作者时间或提交说明可能不同。
比较 `main..feat/nndeploy-pipeline` 需要该功能分支已经存在。

## 8. 撤销暂存、恢复文件和回退已提交改动

撤销某个文件的暂存，保留工作目录内容：

```powershell
git reset HEAD -- docs/git-commands.md
```

丢弃某个已跟踪文件尚未暂存的修改，恢复为暂存区中的内容：

```powershell
git diff -- src/deepsort.cpp
git checkout -- src/deepsort.cpp
```

第二条会丢弃该文件未暂存的改动，先查看差异。已经暂存的修改仍会保留。

撤销一个已提交版本的效果，以一个新的提交记录撤销结果：

```powershell
git revert COMMIT_SHA
```

把 `COMMIT_SHA` 替换成需要撤销的真实提交 SHA；若发生冲突，需要先解决冲突。

## 9. 本次特殊操作：对齐连接工具上传后的提交

本地首个提交为 `7eaf1c9`，通过 GitHub 连接建立的完整基线提交为 `7e91fc8`。
上传后逐项确认文件树相同、工作目录无修改，再保存本地原提交并对齐 `main`。

下面是本次操作的历史记录，日常同步使用第 5 节的流程即可：

```powershell
git rev-parse 'HEAD^{tree}'
git rev-parse 'origin/main^{tree}'
git diff --exit-code HEAD --
git branch backup/local-initial 7eaf1c97bf038ecebd465981f63e7da488be781e
git update-ref -m 'Align baseline with GitHub connector upload' refs/heads/main 7e91fc80d87c4a506d4721e90ccb8e1e800b4ef3 7eaf1c97bf038ecebd465981f63e7da488be781e
git branch --set-upstream-to=origin/main main
```

`update-ref` 移动本地分支指针；末尾的旧 SHA 用来确保分支仍位于预期提交。
它不会改写工作文件。本次使用前已核实两边文件树相同，原提交保存在 `backup/local-initial`。

## 每次开发的常用顺序

查看状态 → 同步 `main` → 创建功能分支 → 修改与验证 → 暂存 → 查看暂存差异 → 提交 → 上传 → PR。

项目文件管理规则和自动测试说明见 [GitHub 项目管理](github-workflow.md)。
