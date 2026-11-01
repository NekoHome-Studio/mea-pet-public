# MeaPet PC 发展建议

> 来源：2026-10-24 现状评估与发展路线图（纯只读审计，基线 `c280ba4`）。
> 本文取其仍有效的结论，并按当前 HEAD `f45fbe9`（2026-11-01 核对）标注已落地项。
> 原则：**已落地的不再重复开工；每项上线必须带可复现的度量读数**。

## 0. 战略判断（仍然成立）

接入面（直连 4 协议 / 3 种 Agent / MCP / OneBot / dsh relay / Live2D）已经足够宽，**未来 6 个月冻结新增接入协议**，把预算转到四个"手感"层面：

1. **她进得去你已有的工具链**（Agent Link / MCP / QQ——这是护城河，别再扩协议）；
2. **她记得**（语义记忆 + 可见可改）；
3. **她先开口**（主动性 + 不打扰）；
4. **她一直在**（分发体积小、装完就能跑、升级不折腾）。

## 1. 审计后已落地的部分（核对至 f45fbe9，勿重复开工）

| 审计结论 | 状态 | 落地 |
|---|---|---|
| 打包版 torch `WinError 1114` 根因（`PyQt5\\Qt5\\bin` 四份 2018 CRT 毒化进程）"未被仓库处理" | ✅ 已修 | `09b97a2`：`MeaPet.spec` 的 `QTBIN_CRT_FILTER` 摘除 4 份旧 CRT，`tests/test_windows_package_shape.py` + `test_vits_deps_probe.py` + doctor 探针钉住 |
| 包体 1.6 GB 需要瘦身 | 🟡 部分 | `09b97a2` 已瘦身 290 MB（spec excludes 收紧）；**"core ≤ 400 MB + 语音包按需下载"未做**，见 P0-2 |
| doctor 在 GBK 控制台崩 | ✅ 已修 | `09b97a2` |

## 2. P0 · 2–4 周（低风险、高收益、可独立验收）

| # | 事项 | 剩余工作 | 验收判据 |
|---|---|---|---|
| P0-1 | 构建后置闸门 | CRT 摘除与形状测试已落地；**缺"构建后冒烟"**：`scripts/build_windows.ps1` 目前只有构建前闸门（LFS pointer、依赖检查） | 构建后跑 `dist\\MeaPet\\MeaPet.exe --doctor` 冒烟，判据写进构建日志，失败即 fail，不产出半成品 dist |
| P0-2 | 发行瘦身 + 分包 | 在已瘦身 290 MB 基础上继续：拆 `core`（PyQt5+网络+UI）/ `voice-local`（torch+VITS）/ `gsv-weights` / `ja-dict`；核对 `llvmlite`/`transformers`/`av` 调用方后 exclude | 默认包 ≤ 400 MB；语音包在向导内按需下载（复用 LFS pointer 检测与 `auto_install_deps`） |
| P0-3 | 产物与更新 | 从零做：Release workflow 产出 `zip + sha256 + 版本号 + 变更日志`；更新用 tufup（TUF 签名增量）或 WinSparkle，**只做"有新版"提示，不静默升级** | 发版产物可校验、可追溯；用户有升级路径 |
| P0-4 | TTS 句级流水线 | 整段合成完才出气泡的结构未动：段内按 `。！？…；` 切句 → 逐句合成 → 首句就绪即出气泡开播，后续句入队；切后端/取消时清队列 | 首字到首音 p50 < 1.2 s（进程内 VITS）；气泡时长规则（max(最短时长, 音频+500ms)）不被破坏 |
| P0-5 | 质量门渐进收紧 | ruff 扩到 `E,F,W,I,B`（分批 + 临时 noqa，禁止一次性 `--fix` 全仓）；CI 加 Python 3.13（Linux）；coverage 先出报告不设阈值；pre-commit（ruff/大文件/LFS pointer/尾随空格） | CI 变红即回退该批规则，不中断主干 |

## 3. P1 · 1–3 个月（产品差异化）

| # | 事项 | 要点 | 验收判据 |
|---|---|---|---|
| P1-1 | 语义记忆 | ONNX Runtime + `bge-small-zh`/`multilingual-e5` 可选本地 embedding；词面检索保留为 fallback；**断网/无模型必须可降级**；设置里加"她记得的事"面板（看/改/删/"忘掉这条"） | 20 条"上周我说过…"检索题 Top-3 命中率 > 80%；降级路径不崩 |
| P1-2 | 主动性 v1 | 本地规则引擎（计划/提醒 + 时段问候 + 久未互动搭话 + 静默时段 + 全局开关），复用分段协议与现有 `agent_proactive` 通道 | 静默时段 0 条主动消息；用户输入期间 0 抢占；每条主动消息可溯源到规则 |
| P1-3 | 语音输入闭环 | sherpa-onnx 流式识别器已有，补 VAD 端点检测 + KWS 唤醒词 + 按住说话 | 8 小时误唤醒 < 1 次；语音回合端到端 < 3 s |
| P1-4 | TTS 引擎插拔 | 从 `MeaTTS(TtsMimoMixin, TtsGsvMixin, TtsVitsMixin)` 抽出 `TtsEngine` 接口 + 能力矩阵（流式/克隆/语言/情绪/体积）；候选：CosyVoice 3（低延迟）、Qwen3-TTS（中文）、Kokoro/Piper（不依赖 torch 的轻量档，同时是 P0-2 瘦身的后续最优解） | 新增引擎不改 `service.py` 以外核心；每引擎留"人耳 + ASR 复读检测"验收记录 |
| P1-5 | 角色包格式 | persona + 表情映射 + 立绘/Live2D + 声音 + 互动语音打成 `.meapetpack`（zip + manifest + 版本 + 签名），向导内导入 | 不改代码导入一个新角色；破坏性字段有版本号与拒绝策略 |

## 4. P2 · 3–12 个月（战略下注）

| # | 事项 | 说明 |
|---|---|---|
| P2-1 | Qt6 / PySide6 spike（2 周，不承诺迁移） | 验证透明置顶窗、托盘、多屏、HiDPI、Wayland layer-shell、QMediaPlayer。触发条件：Qt5 出影响用户的安全问题，或 Python 3.14 不再有 PyQt5 wheel |
| P2-2 | 跨端 | 与 MeaPet Android 共用 Agent Link 协议；人格/记忆同步（可选 E2E 加密或局域网直连） |
| P2-3 | 多 Agent 编排 + MCP 双向 | 桌宠按能力路由到多个 Agent；同时作为 MCP client 挂外部工具；A2A 兼容评估放最后 |
| P2-4 | i18n 与可访问性 | UI 字符串抽表（至少 zh/en）；大字号、色盲友好、界面语言与 TTS 语言解耦 |

## 5. 度量基线（写进 CI 或发版检查表）

| 指标 | 现状 | 目标 |
|---|---|---|
| 默认发行包体积 | 1.6 GB（已瘦身 290 MB，需重测） | ≤ 400 MB（core） |
| 增量更新包 | 不存在 | ≤ 30 MB |
| 首字到首音（VITS 进程内） | 整段合成，秒级 | p50 < 1.2 s |
| 冷启动到可交互 | 未量化 | < 2.5 s |
| 记忆检索 Top-3 命中率 | 未量化（词面） | > 80%（20 题固定集） |
| 唤醒词误触发 | 无此功能 | < 1 次 / 8 h |
| dist 构建冒烟 | 无（仅构建前检查） | 每次构建必跑 `--doctor` |

> 注：体积行以 2026-10-05 的 dist 快照为基线，spec 已改两轮，执行 P0-2 前先重测。

## 6. 风险登记册（精简）

| 风险 | 概率/影响 | 缓解 |
|---|---|---|
| 打包版再次"起不来/假绿" | 高/高 | P0-1 后置闸门 + 冒烟判据 |
| 单维护者倦怠 | 中/高 | 冻结接入面；质量门自动化；角色包让社区分担内容 |
| Qt5 安全事件 / 新 Python 无 wheel | 中/高 | P2-1 spike 提前准备；保留"PNG + 系统字体"兜底 |
| MCP / OneBot / relay 三个网络面攻击面 | 中/中 | 已有 Bearer/mTLS/白名单；补 threat model，"默认关"写进配置默认值 |
| `chat_history` 只标 `summarized=1` 永不删，无限增长 | 高/低 | 加保留期：摘要已覆盖的原始行按天数清理 |

## 7. 必须保护的资产（不要为了"现代化"拆）

1. **契约文档**：`docs/agent-link-v1.md`、`docs/backend-and-control.md`、`docs/onebot-architecture.md`。
2. **诊断文化**：`--doctor` + 探针脚本 + "判不了就说判不了"的措辞纪律。
3. **平台划界测试**：Linux 专属判据显式钉 `sys.platform`，非 Linux 走 skip。
4. **隐私设计**：截图每次确认、仅内存、不落盘；云端视觉需显式 `watcher.allow_cloud`；MCP 默认只监听回环。
5. **中国市场适配**：清华源默认、MiMo 联动、OneBot/QQ 双向、中英双 README。

## 8. 执行节奏

- P0 全部做完之前，不新增接入协议。
- 每个 P1 项上线必须带第 5 节的一个可复现读数，否则不算完成。
- `tts/`、`memory/` 的深改（P0-4、P1-1）排期前确认同仓库没有其他正在进行的改动，避免互相覆盖。
