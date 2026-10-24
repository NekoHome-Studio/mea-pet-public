# -*- mode: python ; coding: utf-8 -*-
"""MeaPet onedir PyInstaller spec (portable data under _internal)."""

import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# 平台分档只用这一个判据，两处（桥接层 / fidus）共用：写三遍 `startswith` 就是三处
# 可以互相不一致的地方。
IS_LINUX = sys.platform.startswith("linux")

# live2d-py package data (native resources); project tree live2d/ is listed in datas.
try:
    live2d_pkg_datas = collect_data_files("live2d")
except Exception:
    live2d_pkg_datas = []

# layer-shell 桥接（Linux / Niri 点击穿透）。它是 ctypes 直读的裸 .so，不是 Python
# 扩展模块，也没有任何 Python 侧 import 指向它——依赖分析**看不见**它。不写这几行，
# 冻结版就没有 Wayland 那条路，而且这条缺失只在真机上暴露，离线测试一律绿。
# 落点 "." = _internal，与 meapet/desktop/wayland_layer.py::shim_candidates()
# 的第一条候选（打包目录）对齐；文件名一致性由 tests/test_layer_shim_packaging.py 钉住。
LAYER_SHIM_NAME = "liblayer_shell_shim.so"
_layer_shim = Path(SPECPATH) / LAYER_SHIM_NAME
if _layer_shim.is_file():
    layer_binaries = [(str(_layer_shim), ".")]
elif IS_LINUX:
    raise SystemExit(
        f"缺 {LAYER_SHIM_NAME}：Linux 打包前先跑 `bash build_layer_shell.sh`"
        f"（产物应落在 {SPECPATH}）。带着缺件打出来的包没有穿透模式。"
    )
else:
    layer_binaries = []

# fidus（切换点定位）随包分发的**平台分档**（2026-09-28 人工 standing 裁决）：
#   Linux   ⇒ 必须在场。wheel 走发行渠道；没装上＝打包缺陷 ⇒ 拦构建，别让缺件出厂。
#   非 Linux ⇒ **显式排除**，直到 fidus 宣布支持该平台。
# "排除"必须写出来，不能靠"这台构建机恰好没装"：眼下 Windows 侧确实是空场，那份"不带"
# 是构建机环境的**副产品**——谁哪天装了个来源不明的 fidus，它就被静默收进 Windows 产物，
# 而那条路在非 Linux 上没有已知的可用引擎。写成 excludes 之后，"不带"是一个可被测试钉住、
# 也可被将来一句话撤销的**决定**。
# 运行面不受影响：`fidus_position.have_engine()` 只在菜单「定位与穿透」那一刻查 ⇒
# 排除掉的效果是那个开关响亮拒绝，不影响开工。
fidus_excludes = [] if IS_LINUX else ["fidus"]

# ── Qt bin 旧 CRT 过滤（2026-10-24 根因实验钉死）────────────────────────────
# PyQt5 的 wheel 会往 Qt5\bin 里塞 4 个 MSVC 运行时（本构建环境实测 14.26.28720.3，
# 2018 年 vcwrkspc 构建）。冻结进程的装载顺序是 PyQt5 先、VITS torch 后：Qt5Core 的
# 静态依赖把这份 14.26 msvcp140 先装进进程，之后 torch 的 c10.dll（VS2022 工具链）
# 在这份 2018 CRT 的进程环境里 DllMain 初始化失败 ⇒ WinError 1114，进程内 VITS
# 整条死。判别实验（9 模式同机对照）：凡进程内 MSVCP140 是 14.26 的场合 torch 必死，
# 换包根 14.44 / 系统 14.51 / miniconda 14.44 全绿——唯一复现 1114 的就是 Qt bin 那份。
# Qt 官方自 5.15.2 起不再往 bin 放 CRT（QTBUG-85833），所以这份旧件本就不该在。
# 删掉后 Qt5Core 的依赖回落到 _internal 根的 14.44 全套（含 _1/_ATOMIC_WAIT），
# 向上兼容，干净机器上也不缺件。
# 只按 (路径前缀, 文件名集合) 精确匹配，Linux 产物（路径分隔符、文件名都不同）零影响。
QTBIN_CRT_FILTER = frozenset(("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"))
PYQT5_QTBIN_PREFIX = "PyQt5/Qt5/bin/"

def filter_qtbin_crts(binaries):
    r"""把 PyQt5\Qt5\bin 里那 4 个旧 CRT 从 Analysis.binaries 摘掉（跨平台安全）。"""
    kept = []
    for name, *rest in binaries:
        norm = str(name).replace("\\", "/")
        base = norm.rsplit("/", 1)[-1].lower()
        if norm.startswith(PYQT5_QTBIN_PREFIX) and base in QTBIN_CRT_FILTER:
            continue
        kept.append((name, *rest))
    return kept

# ── 非 Linux 侧的瘦身排除清单（2026-10-24 依赖图实测）────────────────────────
# 拖整条重链进图的根是 meapet/tools/gsv_infer.py:73 函数体内那句 `import torchaudio`
# （POSIX sox 处置，Windows 冻结进程永不执行，脚本本身只落盘给外部解释器）。
# modulegraph 不分函数体内外，于是 torchaudio→av→transformers/pandas/sklearn/
# numba/llvmlite/librosa 全家被收进包。产品代码（meapet/wizard/pet.py）对这些包
# 零直接 import（xref 反查 + grep 双证）；av 更是仓库注释里写明的死需求。
# Linux 侧不排除：fidus 那格之外，linux_requirements.txt 明列 PyOpenGL，Live2D
# 渲染链也在场，不做"这台机器恰好没有"的赌注——瘦身是 Windows 包的事。
VITS_TRIM_EXCLUDES = [
    "av",
    "librosa",
    "llvmlite",
    "numba",
    "pandas",
    "sklearn",
    "torchaudio",
    "transformers",
]
heavies_excludes = [] if IS_LINUX else list(VITS_TRIM_EXCLUDES)
if IS_LINUX:
    if importlib.util.find_spec("fidus") is None:
        from meapet.bootstrap import fidus_install_hint

        raise SystemExit(
            "本环境没有 fidus ⇒ 这个包不会有「切换点定位」这条能力\n"
            "（Linux 打包前先把它装上）：\n" + fidus_install_hint()
        )
else:
    print(
        "[MeaPet.spec] fidus 按裁决不进非 Linux 包（显式 excludes，撤销见本文件注释）",
        file=sys.stderr,
    )

a = Analysis(
    ["pet.py"],
    pathex=[],
    binaries=filter_qtbin_crts(layer_binaries),
    datas=[
        ("live2d", "live2d"),
        ("sprites", "sprites"),
        ("models/GPT_weights", "models/GPT_weights"),
        ("models/SoVITS_weights", "models/SoVITS_weights"),
        ("GPT-Sovits", "GPT-Sovits"),
        ("config.example.json", "."),
        ("vits_models", "vits_models"),
        ("vits_core", "vits_core"),
        ("vits_requirements.txt", "."),
        ("dic", "dic"),
        ("meapet/assets", "meapet/assets"),
        ("meapet/tools", "meapet/tools"),
    ]
    + live2d_pkg_datas,
    hiddenimports=[
        # certifi CA bundle (belt-and-suspenders; shared client uses OS trust store)
        "certifi",
        "pkg_resources",
        # Agent WebSocket 传输：ws_transport 顶层导入 websockets.exceptions，
        # 连接时才 import websockets.asyncio.client。漏打这些会让依赖门禁
        # 判定 Agent 模式不可用，冻结版曾因此直接静默退出。
        "websockets",
        "websockets.exceptions",
        "websockets.asyncio.client",
        # Companion MCP 服务（agent_control.enabled 时按需导入）
        "mcp",
        "mcp.server",
        "mcp.server.session",
        # wizard package (dynamically imported from _reopen_setup_wizard)
        "wizard.app",
        "wizard.pages",
        "wizard.styles",
        "wizard.platform_info",
        "wizard.connection_test",
        "wizard.env_utils",
        "wizard.widgets",
        "wizard.page_env",
        "wizard.page_llm",
        "wizard.page_backend",
        "wizard.page_tts",
        "wizard.page_tts_vits",
        "wizard.page_tts_gsv",
        "wizard.page_tts_mimo",
        "wizard.page_vision",
        # desktop modules loaded dynamically
        "meapet.desktop.live2d_widget",
        # live2d-py 只认 1.0 这一代（人工 2026-10-17 裁决：不背两代兼容债，安装面钉 >=1.0.0）。
        # 这一格列 `live2d` 是因为桌宠**不静态 import 它**——`meapet/live2d_api.py` 在 try 里
        # import，PyInstaller 的静态图对"闸门 + 动态加载"这条组合不放心，显式列一次最省。
        # 旧名 `live2d.v3` 已从这里删掉：v1.0.0 发布正文写着「Removed the live2d.v3 and
        # live2d.v2cpp modules」，留着它只会在每次构建多一条 warning，并给下一个人留下
        # "这代还活着"的假印象。
        # `live2d.v2`（1.0.0 wheel 里 96 个 .py 有 88 个在它下面）**不列**，实测没有任何 import
        # 边：wheel 内无一处 `import live2d.v2`，`live2d/__init__.py` 唯一的 import 是
        # `from ._live2d import *`，而 `_live2d.so` 里 `PyImport_Import*`/`import_module`
        # 两种字符串各 0 次（原生码不回头 import Python 模块）。原生 `init()` 只是把包目录
        # 交给 `init_internal` 去读着色器资源，那是上面 `collect_data_files("live2d")` 的活。
        "meapet.live2d_api",
        "live2d",
        "meapet.desktop.status_panel",
        "meapet.desktop.timeline_viewer",
        "meapet.desktop.chat_input",
        "meapet.desktop.dialogs",
        # agent adapters
        "meapet.agent.agent_link",
        "meapet.agent.openclaw",
        "meapet.agent.hermes",
        # doctor 那一格 `--agent-script` 用的假 Agent 端。它**不在** `tests/`（PyInstaller 不收
        # 那目录），而是在产品侧 `meapet/agent/`，所以冻结包里也得带上；它走的是 websockets 的
        # **服务器**侧，上面那三行只带了 client——漏了这一行，冻结版跑到那一格是
        # ModuleNotFoundError，不是读数。
        "meapet.agent.fake_agent",
        "websockets.asyncio.server",
        # in-process VITS
        "meapet.tts.engines.vits_runtime",
        "scipy",
        "scipy.io",
        "scipy.io.wavfile",
        # vits_core 文本前端的语言支线（vits_core 是 datas 落盘，PyInstaller 的
        # 依赖分析不扫它——不显式列就会像 2026-10-24 之前的包那样，进程内路
        # 死在 ModuleNotFoundError: unidecode）。清一色**模块级** import
        # （vits_core/text/cleaners.py 顶部一次全量收），缺一个全链死，包括 zh。
        # pyopenjtalk **不在此列**：其 -prebuilt 轮子只到 cp311（vits_requirements.txt
        # 顶部注释的现量），3.13 构建环境装不上 ⇒ 3.13 打的包里 ja 前端整支缺席，
        # 进程内路在包内仍不可用（应配 tts.vits_python 走子进程）；哪天上游发了
        # cp312+ 轮子，加一行 "pyopenjtalk" 即收编。
        "unidecode",
        "inflect",
        "eng_to_ipa",
        "jamo",
        "ko_pron",
        "pypinyin",
        "jieba",
        "cn2an",
        "opencc",
        "indic_transliteration",
        "num_thai",
        "regex",
        # misc dynamic imports
        "translators",
        "jieba",
    ],
    hookspath=["."],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "test",
        "unittest",
        "pydoc",
        "doctest",
    ] + fidus_excludes + heavies_excludes,
    noarchive=False,
    optimize=0,
)
# a.binaries 是 PyInstaller 依赖分析的结果（比 layer_binaries 大得多）：Qt bin
# 的旧 CRT 就混在里面，必须在交给 PYZ/EXE/COLLECT 之前摘掉。
a.binaries = filter_qtbin_crts(a.binaries)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MeaPet",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MeaPet",
)
