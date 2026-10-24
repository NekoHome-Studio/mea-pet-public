# MeaPet 常见问题排查

本页适用于源码运行和 Windows PyInstaller onedir 发行包。先确认程序使用的
配置文件和 Python 环境，再按对应功能排查。完整配置说明见
[README](../README.md) 和 [后端与控制文档](backend-and-control.md)。

## 提交诊断信息前

- 运行日志位于源码根目录的 `logs/`，或 Windows 发行包的
  `MeaPet/_internal/logs/`。
- 只提供与故障时段相关、已经人工检查并脱敏的日志。显式启用 TRACK 调试时，
  日志可能含对话正文。
- 不要上传 `config.json`、API Key、Bearer Token、`mea_memory.db`、截图原图或
  未检查的完整日志目录。
- 同时说明运行方式、操作系统、Python 版本、所选后端和可复现步骤。

## 对话没有回复

1. 在“设置与数据 -> 打开配置页 -> 对话”确认 API 地址、模型名称和 Key。
2. 使用供应商当前文档列出的模型，不要照搬旧版本或第三方教程中的模型名。
3. 本地 Ollama 可先运行 `ollama list`，确认配置的模型已经下载且服务可访问。
4. 检查代理是否只作用于预期供应商，以及系统时间和 TLS 证书是否正常。

## 屏幕识图没有结果

1. 确认屏幕识图不是 `disabled`，并检查当前选择的是 `inherit` 还是 `relay`。
2. 本地 Ollama 模型必须出现在 `ollama list` 中；模型名应与配置完全一致。
3. 云端视觉需要有效凭据。每次截图还必须在本机确认，超时会自动取消。
4. 截图默认只在内存中传递；不要为了排障手动提交含私人内容的截图。

## 没有声音

1. 在“设置与数据 -> 打开配置页 -> 语音”确认 TTS 已启用并测试当前引擎。
2. GPT-SoVITS 需要独立的 Python 运行环境，以及已实际下载的 `.ckpt` / `.pth`
   模型文件；Git LFS pointer 不是可用模型。
3. 多语言参考音频应放在 `voice_cache/` 或使用绝对路径，并确保语言配置匹配。
4. TTS 失败时桌宠会保留文字回复。检查脱敏日志中的引擎状态和错误类型。
5. VITS 走哪条路由日志里的 `VITS route: mode=… reason=…` 决定，只有两种：
   `mode=subprocess`（外部解释器跑 `vits_infer.py`）与 `mode=inprocess`
   （本进程 torch）。这一行就是最终裁决，不要照配置猜。

### 有声音但念不对：先认清保证口径

按 2026-10-13 的人耳验收，本作只保证 **VITS × zh** 的朗读内容正常；其余「引擎 × 语言」组合
（含 GPT-SoVITS 全部语言、VITS 的 ja/en）**只保证出声**，不保证内容正确、完整或可懂。所以
「GPT-SoVITS 念中文念成半句 / 复读」「VITS 念日文听不出是日文」这类现象，在这一格上是已知状态，
不是你的环境坏了。分档明细见
[README 的多语言语音](../README.md#能力口径哪一格真的听得懂)。

反过来说：如果**根本没出声**，那才回到上面这一节的排查项——体检绿灯判的是前置条件齐不齐
（解释器、依赖、权重、参考音频），跟内容对不对无关，两件事别混着查。

### VITS 填了解释器却不走子进程 / 填了 `vits_inprocess: true` 却没生效

`tts.vits_python` 只在它真是一个可用的 Python 解释器时才生效：空串、指向
`MeaPet.exe` 自己、或路径不在盘上，三者都会被判为“没配”，此时打包版回落到
进程内 torch。启动日志对这种情况有一行 `tts.vits_python=… 不是可用的 Python
解释器` 的警告，向导里点保存也会当场提示。

`tts.vits_inprocess: true` 不会否决一个真的外部解释器：外部解释器优先是有意的
（打包版自带的 torch 可能加载不了），此时日志会明确写出
`reason=explicit_inprocess_overridden_by_external`。想让进程内真正生效，先把
`tts.vits_python` 清空。

`tts.vits_model` / `tts.vits_config` / `tts.vits_speaker` 留空即使用随包的
`vits_models/G_latest.pth` 与 `vits_models/finetune_speaker.json`。说话人名字不在
`finetune_speaker.json` 的 `speakers` 表里时会回落到 0 号音色，**并留下一行
warning**，不会静默换音色。

完整判据表、全部配置键与实测读数见
[`docs/tts-vits-routing.zh-CN.md`](tts-vits-routing.zh-CN.md)。

### VITS 进程内报 `WinError 1114`（c10.dll 初始化失败）

症状：日志里 `VITS in-process exception: OSError: [WinError 1114]`，
体检却报绿（或修复后的版本会报红并写明 `torch 寻得到但加载不起来`）。

根因（2026-10-24 判别实验钉死）：PyQt5 的 wheel 往 `Qt5\bin` 里塞了一份
2018 年的 MSVC 运行时（14.26）。冻结进程里 PyQt5 先加载，Qt5Core 把这份旧
`msvcp140.dll` 先装进进程；之后 VITS 懒加载 torch 时，`c10.dll`（VS2022
工具链构建）在这份 2018 CRT 的进程环境里 DllMain 初始化失败 ⇒ `WinError 1114`。
同机九模式对照实验：凡进程内 MSVCP140 是 14.26 的场合 torch 必死；换成包根
14.44 / 系统 14.51 全绿。这份旧 CRT 是 Qt 5.15.2 起官方就不再随包分发的
（QTBUG-85833），本不该出现在冻结包里。

处置：2026-10-24 起 `MeaPet.spec` 的 `filter_qtbin_crts` 会在构建时把这 4 个
旧 CRT 从 `PyQt5\Qt5\bin` 摘掉，Qt5Core 的依赖自然回落到 `_internal` 根的
新版（向上兼容）。**用此之后的构建包不再出现此故障**；旧包的绕行方案是配
`tts.vits_python` 指向一个装了 torch 的真解释器（子进程路线不加载 PyQt5，
不受影响）。

### VITS 进程内合成报 `ModuleNotFoundError`（unidecode / cn2an 等文本前端包）

2026-10-24 前的包还叠着第二层问题：`vits_core` 虽然整目录落盘随包，但它的
**文本前端依赖**（unidecode/cn2an/pypinyin/opencc/inflect/eng_to_ipa/jamo/
ko_pron/indic_transliteration/num_thai）没有被 PyInstaller 收进包——
`vits_core` 走 datas 通道，依赖分析不扫它。修完 CRT 后这层就会露头：
torch 加载成功、`import text` 阶段死在 `ModuleNotFoundError: unidecode`。

2026-10-24 起 spec 把这 10 个包列进 hiddenimports，构建环境装上即随包。
**pyopenjtalk 是已知例外**：其 `pyopenjtalk-prebuilt` 轮子只发到 cp311，
Python 3.13 的构建环境装不上；而 `vits_core/text/cleaners.py` 是模块级
全量 import，ja 前端缺席 ⇒ 进程内路整条不可用（连 zh 句也会死，因为
cleaners 顶部先 import 日文支）。所以 **3.13 构建的包里进程内 VITS 仍不可用**，
正确姿势是配 `tts.vits_python` 指向 3.10/3.11 建的 VITS 环境走子进程路
（`vits_requirements.txt` 顶部有同款说明）；上游发 cp312+ 轮子后此边界消失。

## 气泡不跟着桌宠（平铺桌面）

在平铺式窗口管理器下（实测环境：Linux + niri），回复气泡不会贴着桌宠出现，而是落在屏幕
正中；此后无论桌宠怎么移动、程序怎么请求位置，它都停在原地不动。这不是 Qt 后端选择的
问题——`xcb` 与 `wayland` 两条臂用同一把尺子量下来表现一致，都是"没采纳请求位置、按
合成器策略居中"。

两相都在意料之中：

- 不利的一面：气泡不再指着桌宠，桌宠在边上时读起来会脱节。
- 有利的一面：落点固定、不会随请求漂移，回复的文字与语音本身完全不受影响。

程序在这里不去猜"屏幕上的桌宠到底在哪"再补偿一次，因为在平铺桌面上那个位置本来就是
窗口管理器说了算，猜出来的补偿只对某一台合成器成立。浮动窗口管理器（Windows，以及
GNOME/KDE 一类的传统桌面）下气泡照常贴着桌宠，本条不适用于它们。

## 依赖安装失败

1. 确认安装命令使用的是启动 MeaPet 的同一个 Python：
   `python -m pip --version`。
2. 源码模式按 README 对应平台那一节重装依赖：Windows 可双击 `启动桌宠.bat`
   （它会复用或创建 `.venv` 并安装核心依赖），Linux 执行 `python -m pip install -r
   linux_requirements.txt`。
3. 项目面向简体中文用户，依赖安装默认使用清华 TUNA 公共镜像。可通过
   `MEAPET_PIP_INDEX_URL=https://pypi.org/simple` 切换到 PyPI 官方源，也可填写
   其他可信的自定义源。
4. 冻结版程序不能用自身的 `MeaPet.exe -m pip` 安装依赖。缺失的可选依赖或
   模型需要在构建环境准备后重新打包。

仍无法定位时，请在 GitHub Issues 中提供最小复现步骤和脱敏后的相关日志片段。
