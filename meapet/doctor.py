"""`meapet --doctor` —— 不进 GUI 的**只读**取证快照。

人工 2026-10-08 放行这句：「无头模式是给AI用的」「目的是测试，不论是否打包」。三条口径写在这里，
别处不复述：
- **只读**：不写 config、不装东西、不挂载 surface、不建窗口。唯一的外设是读日志尾部与只读的
  import/模块规格探测。（`--with-gui` 会建一个 Qt 应用对象去取屏幕几何与输入阈值那两格——那是显式
  opt-in，默认关着：无显示服务时建 Qt 应用不是抛异常，是 abort，catch 不到。）
  唯一的**写侧**例外是 `--speech-latency`：它真起推理子进程并往 `tts.output_dir` 落 wav
  （合成不播放，所以不出声）。`--speech-repetition` 走的是**同一个消费者**（再落一发 wav），
  另加一道只读外设：加载仓内那台 sherpa-onnx 中文模型把产物转成字面串。默认档一条都不跑，
  因为那一发按引擎是十几秒到两分钟。
  另有一档**外设**例外是 `--probe-agent`：它先问一次 TCP 端口在不在听，在听才真开一次 WebSocket
  去问 gateway 在不在场，且**只连回路**（127.0.0.1/::1/localhost），非回路端点报"未连接"而不去连——
  往外发请求不在这个入口的授权里。默认档同样一次都不发。
  第三档是 `--probe-source URL` / `--probe-post URL --probe-body JSON`（探候选内容源）：真发一次
  GET 或 POST，但同样**默认只连回路**，往回路外发须再显式带 `--allow-egress`——新增外联要人工
  明确点头，这一档就是为那句留的闸。`--probe-family {auto,ipv4,ipv6}` 管那一发走哪一族地址栈，
  读数里另印**实连的对端地址**：本机 IPv6 出口不稳，不分家族的"连不上"读数判不了案（见坑「网络
  取证的"连不上"必须分 IP 家族重测」）。
  第四档 `--agent-script [PATH]` 动的是**反方向**：不往外连，而是就地起一台**只监听 127.0.0.1** 的
  假 Agent 端（`meapet/agent/fake_agent.py`），让产品自己那个客户端适配器连上来，按剧本发几发
  回合外的帧，报"这端搬了几发"。它因此不占 `--allow-egress` 那道闸——起服务器不等于外联，
  而剧本里写地址/会话那类键是直接拒的。默认档同样一台不起。
- **每一项后面直接跟判据**（这条在说什么 / 什么算坏）。否则"这台机器没有 layer-shell"和
  "我们的桥接层没编进包"会被读成同一句话——那正是本入口要消灭的形状。
- **探测项一律调产品侧那一个函数**，这里不重算。重算就是第二套真值，doctor 会开始自证。

退出码：0 = 探测跑完了（**读数坏不坏不影响这个数**，坏法要出现在文本里）；
1 = doctor 自己没跑起来（那种时候没有任何读数，不能与"跑完但一切正常"共用 0）。
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import ipaddress
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:  # 只为那一格的读数排版；运行期这模块不 import 产品侧适配器
    from meapet.agent.base import ToolStatus

DOCTOR_FLAG = "--doctor"

#: 只打这些键：都是布尔/枚举档，值本身不携密。密钥面（api_key 等）压根不进白名单。
CONFIG_SUMMARY_KEYS = (
    ("live2d", "enabled"),
    ("fidus", "enabled"),
    ("voice_input", "enabled"),
    ("vision", "mode"),
    ("tts", "enabled"),
    ("tts", "engine"),
    ("tts", "voice_lang"),
    ("tts", "timeout"),
)


@dataclass(frozen=True)
class Item:
    """一行读数：名字、值、以及"这条在说什么"。"""

    name: str
    value: str
    judge: str


def doctor_requested(argv: object = None) -> bool:
    """这条命令是不是来取证的。**整词匹配**——`--doctors` 那种参数不该被劫持成取证。"""
    return DOCTOR_FLAG in [str(arg) for arg in (argv if argv is not None else sys.argv)]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="meapet --doctor",
        description="不进 GUI 的只读环境快照（给 AI / 维护者取读数用）",
    )
    parser.add_argument(DOCTOR_FLAG, action="store_true", help="跑取证并退出")
    parser.add_argument(
        "--doctor-format",
        choices=("text", "json"),
        default="text",
        help="text=人读的一行一项；json=机器读（stdout 只有 JSON，其余串到 stderr）",
    )
    parser.add_argument(
        "--config", type=str, help="指定 config.json 路径（默认走产品那一条 config_path()）"
    )
    parser.add_argument(
        "--with-gui",
        action="store_true",
        help="建一个 Qt 应用对象以取屏幕几何（无显示服务时可能直接 abort，默认不开）",
    )
    parser.add_argument(
        "--skip-tts", action="store_true", help="跳过两路 TTS 前置探测（那两格最慢，各跑一次 health_check）"
    )
    parser.add_argument(
        "--skip-layer", action="store_true", help="跳过 layer 桥接层探测（它会真去 init 一次）"
    )
    parser.add_argument(
        "--speech-latency",
        action="store_true",
        help="每路引擎再各真合成两发并报逐发秒数（本入口唯一的写侧：起推理子进程 + 落 wav，不播放）",
    )
    parser.add_argument(
        "--speech-repetition",
        action="store_true",
        help="每路引擎各真合成一发**中文**并用仓内 ASR 转写，报「字种/最长单字连读」两列"
             "（复读崩环初筛；除落 wav 外还加载一次只读 ASR 模型，两档都要显式带）",
    )
    parser.add_argument(
        "--probe-agent",
        action="store_true",
        help="额外探一次 agent 通道（gateway）在场性：先问 TCP 再真连 WebSocket，**只连回路**，非回路端点不连",
    )
    parser.add_argument(
        "--agent-script",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="就地起一台**只监听 127.0.0.1** 的假 Agent 端，按剧本往产品那个客户端适配器发几发"
        "回合外帧，报「落队 N/M」等计数（测的是这端那段搬运，不是他那条真端点）。"
        "不带 PATH 用内置剧本，带 PATH 读那份 JSON（剧本里写地址/会话键一律拒）",
    )
    parser.add_argument(
        "--probe-source",
        action="append",
        default=[],
        metavar="URL",
        help="探一个候选内容源（点击反馈那两行的取证）：GET，逐请求秒级超时，只印状态/字节/首个可见文本片段；可重复",
    )
    parser.add_argument(
        "--probe-post",
        action="append",
        default=[],
        metavar="URL",
        help="同 --probe-source，但发 POST，请求体取 --probe-body（每一发 POST 共用那一份，要分别传就分次跑）；可重复",
    )
    parser.add_argument(
        "--probe-body",
        default="",
        metavar="JSON",
        help="配合 --probe-post：JSON 字符串。体原文会跟着读数一起印——延迟几秒必须连着"
        "「用什么查询打出来的」才读得出（人工 2026-10-08 那句「标注使用的配置」）",
    )
    parser.add_argument(
        "--probe-header",
        action="append",
        default=[],
        metavar="K:V",
        help="给内容源探针追加请求头（Tavily 的 X-Tavily-Access-Mode / Authorization 走这里）。"
        "**值一律不进读数**；但它会出现在本进程的 argv 里，同机其他用户看得见——凭据用环境变量带进来",
    )
    parser.add_argument(
        "--probe-family",
        choices=("auto", "ipv4", "ipv6"),
        default="auto",
        help="内容源探针走哪一族地址栈：auto=与产品同路径（双栈由系统挑），ipv4/ipv6=强制那一族。"
        "本机 IPv6 出口不稳，所以「连不上」这一档必须能分族重测才判得了案",
    )
    parser.add_argument(
        "--allow-egress",
        action="store_true",
        help="配合 --probe-source/--probe-post：允许真往回路外发请求。**新增外联**，须人工明确点头后才带这个",
    )
    return parser.parse_args(argv)


def _probe(name: str, judge: str, thunk) -> Item:
    """探测**自身**的失败也要变成一行读数，而不是 traceback——无头那三个字判的就是这一格。"""
    from meapet.utils import redact_text

    try:
        value = thunk()
    except Exception as exc:  # noqa: BLE001 - 取证面不许把异常递给调用方
        value = f"探测自身失败 {type(exc).__name__}: {exc}"
    return Item(name=name, value=redact_text(str(value)), judge=judge)


def _runtime() -> str:
    from meapet import __version__
    from meapet.paths import is_frozen

    return (
        f"版本 {__version__} · {platform.system()} {platform.release()} · "
        f"python {platform.python_version()} · sys.platform={sys.platform} · "
        f"frozen={is_frozen()} · exe={sys.executable}"
    )


def _dirs() -> str:
    from meapet.log import LOG_DIR
    from meapet.paths import get_data_dir

    root = get_data_dir()
    return (
        f"数据目录={root}（存在={os.path.isdir(root)}） · "
        f"日志目录={LOG_DIR}（存在={os.path.isdir(LOG_DIR)}）"
    )


def _config_summary(config: dict) -> str:
    from meapet.config.store import config_path, scrub_secrets

    clean = scrub_secrets(config or {})
    parts = []
    for section, key in CONFIG_SUMMARY_KEYS:
        block = clean.get(section)
        value = block.get(key) if isinstance(block, dict) else None
        parts.append(f"{section}.{key}={'未设' if value is None else value}")
    return f"读自 {config_path()} · " + " ".join(parts)


def _llm(config: dict) -> str:
    """对话 HTTP 那一路的只读判据：端点归族 + 模型名 + 密钥在场与否（**永不印值**）。

    `api_base` 原文不印：有人把 token 写在查询串里，打出来就是外送。产品侧
    `llm_endpoint_family()` 已经把"连的是哪一家"归成族名，够用且不携密。

    这格**只覆盖 direct/HTTP 那一路**。agent 链接那一路的凭据在 `meapet/agent/factory.py:20`
    的私有 `_resolve_secret` 加三张各不相同的 env 表里，doctor 复制它就是第二套真值——
    所以读数里明写范围，别让"key=缺席"被读成"这台机器没配任何凭据"。
    """
    from meapet.config.store import llm_endpoint_family, resolve_direct_api_key

    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    direct = llm.get("direct") if isinstance(llm.get("direct"), dict) else {}
    agent = llm.get("agent") if isinstance(llm.get("agent"), dict) else {}
    family = llm_endpoint_family(llm) or "未识别"
    model = str(llm.get("model") or direct.get("model") or "").strip() or "未设"
    in_file = any(
        str(block.get("api_key") or "").strip() for block in (llm, direct, agent)
    )
    resolved = bool(resolve_direct_api_key(llm))
    source = "配置" if in_file else ("环境变量" if resolved else "无")
    return (
        f"family={family} model={model} "
        f"direct/HTTP key={'在场' if resolved else '缺席'}(来源={source})"
    )


def _asr(config: dict) -> str:
    """语音输入的模型在场：按**两档**各问一次产品侧那一个真值。

    分档报是因为产品自己那张表就写着这一怕（`meapet/paths.py` 注释：「下了 fp32 却仍读
    int8」）。只报配置那一档的话，这种坏法读数上看起来"模型没下载"，实际是下错了档。
    """
    from meapet.paths import (
        VOICE_ASR_MODEL_FILES,
        find_voice_asr_model_dir,
        normalize_voice_asr_precision,
        project_path,
        voice_asr_cache_dir,
    )

    vi = config.get("voice_input") if isinstance(config.get("voice_input"), dict) else {}
    chosen = normalize_voice_asr_precision(vi.get("precision"))
    roots = f"{voice_asr_cache_dir()} 与 {project_path('models', 'voice_asr')}"
    parts = []
    for precision in VOICE_ASR_MODEL_FILES:
        found = find_voice_asr_model_dir(precision)
        tag = "配置档" if precision == chosen else "另一档"
        parts.append(f"{precision}({tag})={'在场' if found else '缺席'}")
    return (
        f"enabled={vi.get('enabled')} 配置档={chosen} "
        + " ".join(parts)
        + f" · 找过={roots}"
    )


def _display_env() -> str:
    def _flag(name: str) -> str:
        value = os.environ.get(name)
        return value if value else "缺席"

    return (
        f"QT_QPA_PLATFORM={_flag('QT_QPA_PLATFORM')} · "
        f"WAYLAND_DISPLAY={_flag('WAYLAND_DISPLAY')} · "
        f"DISPLAY={_flag('DISPLAY')} · "
        f"XDG_SESSION_TYPE={_flag('XDG_SESSION_TYPE')}"
    )


def _backend() -> str:
    from meapet.desktop.click_through import platform_backend_name

    return f"platform_backend_name={platform_backend_name()}"


def _layer() -> str:
    from meapet.desktop import wayland_layer

    candidates = list(wayland_layer.shim_candidates())
    present = [path for path in candidates if os.path.exists(path)]
    available = wayland_layer.is_available()
    backend = wayland_layer.get_backend()
    last_error = backend.last_error() if backend is not None else "无后端对象"
    return (
        f"候选 {len(candidates)} 条 / 文件在场 {len(present)} 条 {present[:2]} · "
        f"is_available={available} · last_error={last_error}"
    )


def _fidus() -> str:
    from meapet.desktop.fidus_position import have_engine

    return f"have_engine={have_engine()}"


def _qt_app():
    """建（或复用）一个 Qt 应用对象——屏幕几何与输入阈值那两格共用同一份引导。

    这一句是全入口唯一会碰显示服务的地方，所以它只在 `--with-gui` 后面走：无显示服务时
    建 Qt 应用**不是抛异常，是 abort**，`_probe` 那道兜底接不住。
    """
    from PyQt5.QtCore import QCoreApplication
    from PyQt5.QtGui import QGuiApplication

    if QCoreApplication.instance() is None:
        QCoreApplication([])
    return QGuiApplication.instance() or QGuiApplication(["meapet-doctor"])


def _screens(with_gui: bool) -> str:
    if not with_gui:
        return ("未取（默认不建 Qt 应用：无显示服务时那一下是 abort，不是异常）。"
                "要这份读数就加 --with-gui")
    from meapet.desktop.screen_geometry import available_geometry_for

    app = _qt_app()
    lines = [f"Qt platformName={app.platformName()}"]
    for screen in app.screens():
        geo = screen.geometry()
        lines.append(
            f"  屏幕 {screen.name()}: {geo.width()}x{geo.height()} @({geo.x()},{geo.y()}) "
            f"ratio={screen.devicePixelRatio()}"
        )
    usable = available_geometry_for(None)
    lines.append(
        f"  可用区={None if usable is None else (usable.width(), usable.height(), usable.x(), usable.y())}"
    )
    return " | ".join(lines)


def _input_thresholds(with_gui: bool) -> str:
    """点击链上那两道**门槛**的数：Qt 系统给的和产品自己写死的，并排印出来。

    分成两组是有原因的：「点一下没反应」有两种完全不同的坏法——拖窗标志先立起
    （`startDragDistance` 通常远小于分区门票那 35 px），或者双击的第二次 release 在
    「单击/双击那道延迟」的窗口内就已经把反馈发出去了（提案里那道延迟要跟的是**系统给的这个读数**，
    不是人工随口举例的那个 300 ms——他 2026-10-11 已裁阈值就是 doubleClickInterval）。
    读数不带这两组数就分不出是哪种，所以两边各印各的，不合成一个"看起来正常"。
    """
    if not with_gui:
        return "未取（要建 Qt 应用，与 screens 同档：加 --with-gui）"

    from meapet.desktop.live2d_widget import (
        ZONE_CLICK_MAX_DISTANCE_PX,
        ZONE_CLICK_MAX_HOLD_SECONDS,
    )

    app = _qt_app()
    hints = app.styleHints()
    return (
        f"Qt platformName={app.platformName()} "
        f"doubleClickInterval={hints.mouseDoubleClickInterval()}ms "
        f"startDragDistance={hints.startDragDistance()}px · "
        f"产品门票={ZONE_CLICK_MAX_DISTANCE_PX}px/{ZONE_CLICK_MAX_HOLD_SECONDS}s"
        "（那两个数读自 live2d_widget 的名字，不是这里抄的）"
    )


def _audio_import() -> str:
    import importlib.util

    return f"pyaudio 规格={'在场' if importlib.util.find_spec('pyaudio') else '缺席'}"


def _audio_playback(config: dict) -> str:
    """播放那一路：Qt 首选在这台机器上到底起不起得来、回退走哪条、音量档生不生效。

    这里**真的 import 一次**绑定，因为 `find_spec` 只查得到 `.abi3.so` 在不在，查不到它
    NEEDED 的那几个 Qt5 库被裁没裁——本案那条日志就是撞在这一格上的。
    """
    import importlib
    import importlib.util
    import shutil

    binding = importlib.util.find_spec("PyQt5.QtMultimedia")
    if binding is None:
        head = "绑定=缺席（PyQt5 没装 QtMultimedia）"
    else:
        try:
            importlib.import_module("PyQt5.QtMultimedia")
            head = "绑定=在场 底层=可加载 ⇒ Qt 那一路可用（音量档生效）"
        except Exception as exc:  # noqa: BLE001 - 取证面把不可用当读数报，不当异常抛
            head = f"绑定=在场 底层=加载失败：{type(exc).__name__}: {exc}"

    if sys.platform.startswith("win"):
        tail = "回退=os.startfile（不吃增益）"
    elif sys.platform == "darwin":
        afplay = shutil.which("afplay")
        tail = f"回退=afplay {'在场(' + afplay + ')' if afplay else '缺席'}（不吃增益）"
    else:
        aplay = shutil.which("aplay")
        tail = f"回退=aplay {'在场(' + aplay + ')' if aplay else '缺席'}（不吃增益）"
    audio = (config or {}).get("audio") or {}
    return (
        f"{head} · {tail} · 音量档 tts={audio.get('tts_volume_percent')}% "
        f"sfx={audio.get('sfx_volume_percent')}%（只有 Qt 那一路读这两个数）"
    )


def _input_devices(config: dict) -> str:
    from meapet.voice.engine import VOICE_INPUT_SAMPLE_RATE, list_input_devices

    cfg = (config or {}).get("voice_input") or {}
    devices = list_input_devices(VOICE_INPUT_SAMPLE_RATE)
    return (
        f"指定设备={cfg.get('input_device_index')} 精度={cfg.get('precision')} "
        f"采样率={VOICE_INPUT_SAMPLE_RATE} · 枚举到 {len(devices)} 台"
        f"（可用 {sum(1 for row in devices if row[3])} 台）"
    )


def _tts(config: dict, engine: str) -> str:
    from meapet.tts.diagnostics import speech_facts, tts_for_engine

    started = time.monotonic()
    tts = tts_for_engine(config, engine)
    health = bool(tts.health_check())
    return (
        f"enabled={getattr(tts, 'enabled', None)} engine={getattr(tts, 'engine', None)} "
        f"前置={'齐' if health else '不齐'} 本项耗时={time.monotonic() - started:.2f}s · "
        f"{speech_facts(engine, tts)}"
    )


def _tts_latency(config: dict, engine: str) -> str:
    from meapet.tts.diagnostics import speech_latency, tts_for_engine

    return speech_latency(engine, tts_for_engine(config, engine))


def _tts_repetition(config: dict, engine: str) -> str:
    from meapet.tts.diagnostics import speech_repetition, tts_for_engine

    return speech_repetition(engine, tts_for_engine(config, engine), config)


#: `--probe-agent` 那一发的总预算（秒）。三家适配器自己的 `timeout_seconds` 是内层闸，
#: 这一道是**外层**的：无头取证里任何一格都不该被一个挂住的端点摁住——挂住的读数和"坏"的
#: 读数在调用方那边长得一模一样，那就等于本入口最想说的那句话（什么算坏）没了。
AGENT_PROBE_BUDGET_SECONDS = 8.0


def _is_loopback(host: str) -> bool:
    """这台端点是不是**回路**：只有回路的才允许真连。

    `localhost` 按名字放行（三家出厂地址都是 `127.0.0.1`）；其余一律不连——doctor 的授权里
    没有"替这台机器往外发请求"这一项，新增外联要人工明确点头。
    """
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _capabilities_summary(caps: object) -> str:
    """能力对象里挑几枚**不携密**的键印出来；三家字段名不同，所以按在场拼，不猜结构。"""
    safe_keys = (
        "platform",
        "protocol",
        "protocol_version",
        "agent_name",
        "server_version",
        "streaming",
    )
    parts = [f"{key}={getattr(caps, key)}" for key in safe_keys if getattr(caps, key, None) is not None]
    return f"{type(caps).__name__} " + " ".join(parts)


async def _probe_and_close(adapter) -> object:
    try:
        return await asyncio.wait_for(adapter.probe(), timeout=AGENT_PROBE_BUDGET_SECONDS)
    finally:
        await adapter.close()


def _port_open(host: str, port: int, budget: float = 2.0) -> bool:
    """那个端口在不在听——**这一道是给 WebSocket 那一发当地闸的**。

    `hermes.py:349` 的 `_ensure_connected` 第一件事是查 token，没配凭据时抛的是
    `ValueError("Hermes WebSocket token is required…")`，而 `probe()` 末尾那个
    `except Exception` 把它归进「握手失败」那一句——于是"这台机器没配 agent 凭据"和
    "gateway 挂了"在读数里长得一模一样。端口先问一句就把这两支分开了：没在听 = 真不在场，
    在听而 probe 抛 = 凭据或协议那一档的坏。
    """
    import socket

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    for family, socktype, proto, _, address in infos:
        with socket.socket(family, socktype, proto) as sock:
            sock.settimeout(budget)
            if sock.connect_ex(address) == 0:
                return True
    return False


DEFAULT_PORTS = {"ws": 80, "wss": 443, "http": 80, "https": 443}


def _endpoint_port(url: str) -> int | None:
    """地址里那个端口——**没写出来时按 scheme 兜**，不在这里猜出厂默认端口。

    出厂地址（`config/defaults.py:17`）自带端口，所以绝大多数情况走的是 `split.port`；
    兜底那三行只为"用户自己填了个不带端口的 ws://localhost"这种形状兜着，宁可报
    "按 ws→80 试的"也不假装知道他要连的是哪一格。
    """
    split = urlsplit(url)
    try:
        port = split.port
    except ValueError:
        return None  # 端口那一栏写坏了（字母、超范围）
    if port is not None:
        return port
    return DEFAULT_PORTS.get((split.scheme or "").lower())


def _failure_chain(exc: BaseException) -> str:
    """把 `raise X from Y` 那条链读全——**产品侧把真因藏在 `__cause__` 里**。

    `hermes.py:596` 是 `raise ValueError("…握手失败") from exc`：外层那句永远一样，
    而底下那层才是"没配 token"还是"对端拒连"。读数只印外层就等于把两支又捏回一支，
    所以这里顺着 `__cause__/__context__` 再走一层（只一层，链再长就截断，防环与刷屏）。
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        parts.append(f"← {type(cause).__name__}: {str(cause)[:180]}")
    return " ".join(parts)


def _agent_channel(config: dict) -> str:
    """agent 链接那一路的在场性——**复用产品侧工厂**，不在这里重算凭据、地址或 kind 默认值。

    凭据解析在 `meapet/agent/factory.py` 的私有 `_resolve_secret` 加三张各不相同的 env 表里，
    doctor 抄它就是第二套真值（`llm` 那格的判据里已经写过这一怕）。所以这一格走的是
    `create_agent_adapter_from_config()` + `probe()`：产品怎么连，我就怎么连。
    """
    from meapet.agent.factory import create_agent_adapter_from_config

    # 工厂会往传进去的那份 dict 里塞 session_id/device_id（`_ensure_local_session`），
    # 而 doctor 的只读承诺是对着调用方那份 config 立的——喂它一份副本。
    probe_config = copy.deepcopy(config or {})
    configured = ((probe_config.get("llm") or {}).get("agent") or {}).get("kind")
    kind = f"配置={configured if configured not in (None, '') else '未设(工厂按 hermes)'}"

    try:
        adapter = create_agent_adapter_from_config(probe_config)
    except Exception as exc:  # noqa: BLE001 - 构造坏也是读数，不是 traceback
        return f"{kind} 适配器构造失败 {type(exc).__name__}: {exc}"

    base_url = str(getattr(adapter.config, "base_url", "") or "")
    split = urlsplit(base_url)
    host = split.hostname or ""
    port = _endpoint_port(base_url)
    label = (
        f"{kind} adapter={type(adapter).__name__} "
        f"host={host or '解析不出'} port={port if port is not None else '解析不出'}"
    )
    if not _is_loopback(host):
        return (
            f"{label} · **未连接**：端点不在回路上。"
            "doctor 不替这台机器往外发请求（新增外联须人工明确点头）"
        )
    if port is None:
        return f"{label} · 不在场：地址里取不出端口（未发 WebSocket）"

    # 先问 TCP，再决定要不要发那一发 WebSocket——未在听就不发。这一道闸把
    # "这台机器没配 agent 凭据"与"gateway 挂了"分开（见 `_port_open` 里那段），
    # 顺带省下对一个死端口做 WS 握手的整预算。
    if not _port_open(host, port):
        return f"{label} · 不在场：tcp={host}:{port} 未监听（未发 WebSocket）"

    try:
        caps = asyncio.run(_probe_and_close(adapter))
    except Exception as exc:  # noqa: BLE001 - 不在场是本案要的那条判据，不是异常
        return f"{label} · 不在场：{_failure_chain(exc)}（预算 {AGENT_PROBE_BUDGET_SECONDS}s）"
    return f"{label} · 在场：{_capabilities_summary(caps)}（{AGENT_PROBE_BUDGET_SECONDS}s 内握手完成）"


#: 假端那一趟要**临时摘掉**的凭据 env。`factory._resolve_secret`（`factory.py:30-33`）把这两个
#: 名字排在**任何** config 字面量之前，所以在合成配置里写 `auth_token=""` 拦不住它们——不摘，
#: 客户端会把这台机器上真那枚 token 递进就地起的监听器。摘在 try/finally 里，跑完原样还回去。
AGENT_SCRIPT_TOKEN_ENVS = ("MEA_PET_AGENT_LINK_TOKEN", "MEA_PET_AGENT_TOKEN")


def _counter_note(status: ToolStatus) -> str:
    """把 `chat.tool_status` 那三个可选计数排成 `#1·250ms·40tok`，缺的那一项**不印 0**。

    印 0 会把「Agent 没给」说成「给了但是零」——那一格在界面上是要人看出差别的。
    """
    parts = []
    if status.call_index is not None:
        parts.append(f"#{status.call_index}")
    if status.elapsed_ms is not None:
        parts.append(f"{status.elapsed_ms}ms")
    if status.tokens is not None:
        parts.append(f"{status.tokens}tok")
    return "·".join(parts)


def _agent_script(config: dict, script_path: str = "") -> str:
    """假 Agent 端跑一趟：剧本发帧，读数说**这端的客户端搬了几发**。

    与 `--probe-agent` 那格分工明确：那一格判的是"他那条真端点在不在场"，这一格判的是
    "帧真到了的时候，客户端那段搬运（`agent_link.py:649-655` 入队 → `take_idle_tool_statuses`）
    有没有跑"。端点是就地起在 127.0.0.1 上的假端（`meapet/agent/fake_agent.py`），
    客户端仍走产品工厂那一条构造路径。

    分母由假端按 `state`/`safe_text` 都非空来算（与客户端 `_tool_status_from_payload` 同判据），
    所以「落队 0/4」是搬运坏了，「落队 0/0」是剧本压根没发到——两种坏在读数里不同形。
    「计数」那一格的分母更直接：假端调的就是客户端那个 `safe_counter`，不是抄一份，所以
    「计数 0/2」只可能是解析那一步坏了，不会读成判据漂移。
    """
    from meapet.agent.fake_agent import (
        DEFAULT_SCRIPT,
        FakeAgentScriptError,
        load_script,
        scripted_agent_server,
    )
    from meapet.agent.factory import create_agent_adapter_from_config

    probe_config = copy.deepcopy(config or {})
    llm = probe_config.get("llm")
    if not isinstance(llm, dict):
        llm = {}
        probe_config["llm"] = llm
    agent = llm.get("agent")
    if not isinstance(agent, dict):
        agent = {}
        llm["agent"] = agent
    configured_kind = agent.get("kind")
    # 这一趟走的是 agent_link 那套信封，所以 kind 由本案定，不跟着他配的 hermes/openclaw 飘。
    agent["kind"] = "agent_link"
    agent["auth_token"] = ""
    agent["api_key"] = ""

    try:
        steps = DEFAULT_SCRIPT if not script_path else load_script(script_path)
    except (OSError, FakeAgentScriptError, UnicodeDecodeError) as exc:
        return f"假端未起：剧本读不出来 {type(exc).__name__}: {exc}"
    source = "内置剧本" if not script_path else f"外部 {script_path}"
    label = f"{source} · 配置 kind={configured_kind or '未设'}（本格强制 agent_link）"

    async def _run_once():
        async with scripted_agent_server(script=steps) as server:
            agent["base_url"] = server.url
            adapter = create_agent_adapter_from_config(probe_config)
            try:
                # 握手调的是产品侧那一个函数（`adapter.probe()`），不在这里重演 handshake 逻辑。
                started = time.monotonic()
                await asyncio.wait_for(adapter.probe(), AGENT_PROBE_BUDGET_SECONDS)
                spent = time.monotonic() - started
                result = await server.wait_result(
                    max(0.5, AGENT_PROBE_BUDGET_SECONDS - spent)
                )
                idle = adapter.take_idle_tool_statuses()
            finally:
                await adapter.close()
        return result, idle

    saved = {key: os.environ.pop(key, None) for key in AGENT_SCRIPT_TOKEN_ENVS}
    try:
        result, idle = asyncio.run(_run_once())
    except Exception as exc:  # noqa: BLE001 - 跑不通是本案要的那条读数，不是 traceback
        return f"{label} · 不在场：{_failure_chain(exc)}（预算 {AGENT_PROBE_BUDGET_SECONDS}s）"
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value

    tokens = "·".join(str(length) for length in result.hello_token_lengths) or "无握手"
    codes = "·".join(result.error_codes) or "无"
    first = f"（首帧 {idle[0].state}·{idle[0].safe_text}）" if idle else ""
    counted = [status for status in idle if _counter_note(status)]
    note = f"（{_counter_note(counted[0])}）" if counted else ""
    return (
        f"{label} · 发出 {result.sent} 帧 · 入站 {len(result.received_types)} 帧 · "
        f"落队 {len(idle)}/{result.queueable_sent}{first} · "
        f"计数 {len(counted)}/{result.counter_sent}{note} · 回执 {codes} · "
        f"快照 {result.snapshot_seen} · 屏障 "
        f"{'pong' if result.barrier_acked else '未到齐'} · token 长度 {tokens}"
        f"（预算 {AGENT_PROBE_BUDGET_SECONDS}s）"
    )


#: 内容源探针的**每次操作**超时（秒），同时兼**延迟判定线**——判定线只判不切：超了照样把成功
#: 读数报出来，人要知道的是"这源能用但要 3.2 s"，不是"它在 3.0 s 被我弄死了"（后者会把可用源
#: 误读成不可用）。这一格存在的理由就是**覆盖**客户端那个默认值
#: （`http_async.py:29` 的 `Timeout(120.0, connect=10.0)`）——不显式压它，一次挂站的检索页
#: 就能把整份取证摁住两分钟。
SOURCE_PROBE_BUDGET_SECONDS = 3.0
#: httpx 的 `timeout` 管的是**每次操作**（连接 / 每一读），不是端到端——每秒滴一口的响应
#: 每一读都不超线、整条链却能跑满 7 s 以上仍报 200（`tests/test_doctor.py` 的
#: `test_trickling_response_is_measured_not_killed` 把这个形状钉成了可复跑的凭据）。
#: 所以上面那枚压不住"点一下等多久"，这一枚才是**端到端硬上限**。
SOURCE_PROBE_HARD_CAP_SECONDS = 8.0
#: "首个可见文本"截多长：够人过目认出是不是想要那句即可，不是把整页正文搬进报告。
SOURCE_SNIPPET_CHARS = 80
#: `--probe-family` 到 httpcore `local_address` 的对照：`"0.0.0.0"` 让 socket 建成 AF_INET、
#: `"::"` 建成 AF_INET6（httpcore 自己就这条逃生口写了这一句）。`auto` 不在表里——它走产品那个
#: 共享客户端、由系统双栈挑，所以**auto 那档的失败读数判不了"这源连不上"**（本机 IPv6 出口不稳，
#: 同一个域名 `-4` 200/2.06s、双栈 8s 无响应，是实测踩出来的）。
SOURCE_PROBE_FAMILY_BIND = {"ipv4": "0.0.0.0", "ipv6": "::"}
#: 请求体印进读数时截多长：那一发的"用什么查询打出来的"要过目，但不是把整份 body 搬回来。
SOURCE_PROBE_BODY_CHARS = 160
#: 失败相位的三张表。判据只有一句：**读数里那句「这一发连上了」只许出现在真连得上的那一档**。
#: 2026-10-09 我实测踩到自己这条——`UnsupportedProtocol`（URL 少了协议段）被 `else` 归进"读段"，
#: 读数于是印出「这一发连上了却没在预算内吐完」：一个包都没发出去的请求被写成连上了，那是假证。
SOURCE_PROBE_CONNECT_PHASE = ("Connect", "Proxy")
SOURCE_PROBE_TRANSFER_PHASE = ("Read", "Write")
SOURCE_PROBE_NOT_SENT_PHASE = (
    "UnsupportedProtocol",
    "InvalidURL",
    "LocalProtocolError",
    "PoolTimeout",
)


def _failure_phase(error: str) -> str:
    """把失败读数按异常名归相位：`connect` / `transfer` / `capped` / `not-sent` / `unknown`。

    `capped` 是这一层自己造的（端到端上限到点掐），它**不知道**卡在哪一段——把它当"读段"就是替
    没测到的事下结论。分不开的一律走 `unknown`，宁可只报"不可判"。
    """
    name = error.split(":", 1)[0].strip()
    if name.startswith(SOURCE_PROBE_CONNECT_PHASE):
        return "connect"
    if name.startswith(SOURCE_PROBE_TRANSFER_PHASE):
        return "transfer"
    if name == "TotalTimeout":
        return "capped"
    if name in SOURCE_PROBE_NOT_SENT_PHASE:
        return "not-sent"
    return "unknown"


def _printable_url(url: str) -> str:
    """印出来用的地址：**查询串整条丢掉**——有人把 token 写在查询里（`llm` 那格同样的怕）。"""
    try:
        split = urlsplit(url)
        port = split.port
    except ValueError:
        return "<地址写法不合法>"
    host = split.hostname or ""
    netloc = f"{host}:{port}" if port is not None else host
    return f"{split.scheme}://{netloc}{split.path}"


def _visible_text(body: bytes, content_type: str) -> str:
    """从响应体里取**一段人能看的文字**——三档解析各走各的，取不到就如实说取不到。

    这一格判的是提案 §2 那两行的"可解析"那半条：状态 200 与一堆字节本身不说明取回了内容，
    所以要把首个可见片段印出来给人过目。失败不抛——"解析不出"就是那条源该被划掉的证据。
    """
    if not body:
        return "<空响应体>"
    lowered = (content_type or "").lower()
    try:
        if "json" in lowered:
            data = json.loads(body.decode("utf-8", errors="replace"))
            pairs = _first_strings(data)
            return " ".join(f"{key}={val}" for key, val in pairs) or "<JSON 里没有字符串字段>"
        if "html" in lowered or "xml" in lowered:
            from lxml import html as lxml_html

            tree = lxml_html.fromstring(body.decode("utf-8", errors="replace"))
            for dead in tree.xpath("//script | //style | //noscript"):
                dead.getparent().remove(dead)
            return " ".join(tree.text_content().split())
        return body.decode("utf-8", errors="replace").strip()
    except Exception as exc:  # noqa: BLE001 - 解析不出是读数，不是 traceback
        return f"<解析失败 {type(exc).__name__}: {str(exc)[:60]}>"


def _first_strings(data: object, limit: int = 3) -> list[tuple[str, str]]:
    """JSON 里前几枚字符串值，**带键名**——古诗源返的是结构，光有值认不出哪句是句。"""
    found: list[tuple[str, str]] = []

    def walk(node: object, key: str = "") -> None:
        if len(found) >= limit:
            return
        if isinstance(node, dict):
            for name, value in node.items():
                walk(value, str(name))
        elif isinstance(node, list):
            for value in node:
                walk(value, key)
        elif isinstance(node, str) and node.strip():
            found.append((key or "?", " ".join(node.split())[:40]))

    walk(data)
    return found


async def _fetch_one(method: str, url: str, body, headers, family: str) -> tuple:
    """一发请求（GET 或 POST）。返回 `(status, content_type, 原始字节, 错误串, 对端注记)`。

    两档路径**有意**分开：`family="auto"` 走 `http_async` 那个共享客户端（TLS/UA/重定向/代理都跟
    产品同源），钉了家族的那两档自起一个客户端——共享那一个没法按次换 transport。代价写在
    `_pinned_client` 里（它绕开环境代理，否则钉的是到代理那一段的家族）。

    两种情形都不把客户端那个默认超时递给调用方：`timeout` 是 httpx 的**每次操作**预算（连不上/
    不再吐字节要多久才放手），`wait_for` 才是**端到端**上限（滴流式慢响应也能掐住）。"""
    from meapet import http_async

    status, content_type, raw, error, peer = None, "", b"", "", ""
    own_client = None
    try:
        kwargs: dict = {"timeout": SOURCE_PROBE_BUDGET_SECONDS}
        if headers:
            kwargs["headers"] = headers
        if family == "auto":
            if method == "POST":
                response = await http_async.post_json(url, json=body, **kwargs)
            else:
                response = await http_async.get_json(url, **kwargs)
        else:
            own_client = _pinned_client(family)
            response = await own_client.request(
                method, url, json=body if method == "POST" else None, **kwargs
            )
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        raw = response.content
        #: 必须在关掉客户端之前取——连接一断底层 socket 就没了，那时再问抛 OSError。
        peer = _peer_note(response)
    except Exception as exc:  # noqa: BLE001 - 挂/拒/超时都是取证要的那条判据
        error = f"{type(exc).__name__}: {str(exc)[:120]}"
    finally:
        if own_client is not None:
            await own_client.aclose()
        else:
            await http_async.aclose_client()
    return status, content_type, raw, error, peer


async def _fetch_capped(method: str, url: str, body, headers, family: str) -> tuple:
    """端到端兜底：超 `SOURCE_PROBE_HARD_CAP_SECONDS` 就掐，报成 TotalTimeout 而不是把挂起递给调用方。"""
    try:
        return await asyncio.wait_for(
            _fetch_one(method, url, body, headers, family), SOURCE_PROBE_HARD_CAP_SECONDS
        )
    except asyncio.TimeoutError:
        return None, "", b"", f"TotalTimeout: 端到端 {SOURCE_PROBE_HARD_CAP_SECONDS}s 用完", ""


def _pinned_client(family: str):
    """按家族钉死的那一发用的独立客户端：TLS 上下文与 UA 照 `_new_client` 那一套，只多一枚 local_address。

    两个刻意的偏离，读数里都看得见：① `trust_env=False`——环境里设了代理时 httpx 会连代理而非源站，
    那时 `local_address` 钉的是"到代理那一段"的家族，读数会说 ipv4、实际源站走的是隧道，那一条假证
    比没证更坏；② `verify` 得传给 **transport**（传给 client 会被 transport 参数整体忽略），
    否则这条路径退回 certifi——产品那个共享客户端用 OS 证书store 正是为了绕开打包态 certifi 缺失。
    """
    import ssl

    import httpx

    from meapet import USER_AGENT

    transport = httpx.AsyncHTTPTransport(
        local_address=SOURCE_PROBE_FAMILY_BIND[family],
        verify=ssl.create_default_context(),
        trust_env=False,
    )
    return httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(SOURCE_PROBE_BUDGET_SECONDS),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def _peer_note(response) -> str:
    """那一发**实际连到了哪个地址**——IP 家族之争只有靠这一格才断得清（`--probe-family auto` 时唯一的信息源）。

    取的是 httpcore 挂在响应扩展里的 `network_stream`（`server_addr` = 对端）。取不到就如实说取不到，
    **不猜**：猜来的地址等于没测。
    """
    try:
        stream = response.extensions.get("network_stream")
        peer = stream.get_extra_info("server_addr") if stream is not None else None
        host = peer[0] if isinstance(peer, (tuple, list)) and peer else str(peer)
    except Exception as exc:  # noqa: BLE001 - 对端取不到是读数缺一格，不是 traceback
        return f"peer=未取到（{type(exc).__name__}）"
    if not host or host == "None":
        return "peer=未取到（连接已关或后端不提供）"
    scope = host.split("%", 1)[0]
    return f"peer={scope}·{'ipv6' if ':' in scope else 'ipv4'}"


def _parse_probe_headers(raw: tuple[str, ...]) -> tuple[dict[str, str], str]:
    """`--probe-header K:V` 收成字典；写法不对就返回一句错误，由调用方报"未请求"——**不猜哪半是名字**。"""
    parsed: dict[str, str] = {}
    for row in raw:
        name, sep, value = row.partition(":")
        if not sep or not name.strip():
            return parsed, f"--probe-header 写法不对：「{row}」（要 K:V）"
        parsed[name.strip()] = value.strip()
    return parsed, ""


def _source_probe(
    url: str,
    allow_egress: bool,
    *,
    method: str = "GET",
    body_json: str = "",
    headers: tuple[str, ...] = (),
    family: str = "auto",
) -> str:
    host = urlsplit(url).hostname or ""
    label = _printable_url(url)
    if method == "POST":
        label = f"POST {label}"
    else:
        method = "GET"
    if not allow_egress and not _is_loopback(host):
        return (
            f"{label} · **未请求**：端点不在回路上。带 --allow-egress 才会真往外发，"
            "而新增外联须人工明确点头——这一格默认不发任何一个包"
        )
    header_map, header_error = _parse_probe_headers(tuple(headers))
    if header_error:
        return f"{label} · **未请求**：{header_error}（一个包没发）"
    body = None
    if method == "POST":
        if not body_json.strip():
            return (
                f"{label} · **未请求**：POST 要配 --probe-body——体原文跟着读数一起印，"
                "不替你猜一份查询（猜来的延迟判不了案）"
            )
        try:
            body = json.loads(body_json)
        except json.JSONDecodeError as exc:
            return f"{label} · **未请求**：--probe-body 不是合法 JSON：{exc.msg} 位置 {exc.pos}（一个包没发）"
        if not isinstance(body, dict):
            return f"{label} · **未请求**：--probe-body 得是 JSON 对象（这一族 API 都收对象）"
    started = time.perf_counter()
    status, content_type, raw_body, error, peer = asyncio.run(
        _fetch_capped(method, url, body, header_map, family)
    )
    elapsed = time.perf_counter() - started
    #: 延迟判定与读数同屏：超线的成功仍是成功，但要让人一眼看出它超了点击反馈那条预算线。
    verdict = "" if elapsed <= SOURCE_PROBE_BUDGET_SECONDS else f" · **超 {SOURCE_PROBE_BUDGET_SECONDS}s 预算**"
    if error:
        #: 挂在哪一段决定下一步测什么：**连接段**才轮到地址栈的怀疑（这台机器 IPv6 出口不稳，
        #: auto 那档把 v6 挂写成"这源连不上"就是假证）；**收发段**与栈无关，劝人分族重测是把人
        #: 往错方向支；而**根本没发出去**那一档，两句都不能说。
        name = error.split(":", 1)[0].strip()
        phase = _failure_phase(error)
        if phase == "connect":
            tail = (
                "（双栈由系统挑 → **未定**：分族重测 `--probe-family ipv4` 与 `ipv6` 各跑一遍才判得了这源死活）"
                if family == "auto"
                else f"（{family} 那一条自己挂的，与另一族无关）"
            )
        elif phase == "transfer":
            tail = "（挂在**收发**那一面：连上了才有这一挂，与地址栈无关，分族重测治不了它）"
        elif phase == "capped":
            tail = (
                "（端到端上限掐的，**相位未知**：可能卡在连也可能卡在读——要断言就把每次操作预算压到上限"
                "之下，看它先挂哪一段）"
            )
        elif phase == "not-sent":
            tail = (
                f"（**一个包都没发出去**：{name} 是探测自己拒的，与源站死活、与地址栈都无关——先修 URL/参数）"
            )
        else:
            tail = f"（相位不可判：{name} 不在已知分类里，别按地址家族下结论）"
        return (
            f"{label} · 失败：{error}（{elapsed:.2f}s；每次操作预算 "
            f"{SOURCE_PROBE_BUDGET_SECONDS}s / 端到端上限 {SOURCE_PROBE_HARD_CAP_SECONDS}s）"
            f" · 家族={family}{tail}"
        )
    snippet = _visible_text(raw_body, content_type)[:SOURCE_SNIPPET_CHARS]
    body_note = f" · 体={body_json[:SOURCE_PROBE_BODY_CHARS]}" if body is not None else ""
    return (
        f"{label} · status={status} ct={content_type or '未声明'} bytes={len(raw_body)} "
        f"耗时={elapsed:.2f}s{verdict} · 家族={family} {peer}{body_note}"
        f" · 首个可见文本=「{snippet}」"
    )


def _log_tail(lines: int = 5) -> str:
    from meapet.paths import data_path

    path = data_path("chat_errors.log")
    if not os.path.isfile(path):
        return f"{path} 不存在"
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        rows = handle.read().splitlines()[-lines:]
    return f"{os.path.basename(path)} 尾 {len(rows)} 行 :: " + " ⏎ ".join(rows)


def collect(config: dict, *, with_gui: bool = False, skip_tts: bool = False,
            skip_layer: bool = False, measure_speech: bool = False,
            measure_repetition: bool = False,
            probe_agent: bool = False, probe_sources: tuple[str, ...] = (),
            probe_posts: tuple[str, ...] = (), probe_body: str = "",
            probe_headers: tuple[str, ...] = (), probe_family: str = "auto",
            allow_egress: bool = False, agent_script: str | None = None) -> list[Item]:
    """跑一遍探测，返回逐项读数（任何一项坏都会变成一行文本，不抛）。

    带 `measure_speech` / `measure_repetition` / `probe_agent` / `probe_sources`|`probe_posts`
    那几档时才跑会动手的格
    （起子进程落 wav / 真开一次 WebSocket / 真发一次 GET 或 POST）；默认三档都不动，一格不碰、
    一个包不发。`probe_sources` 与 `probe_posts` 只连回路，除非同时显式带 `allow_egress`
    （那是"新增外联"，须人工点头）。

    `agent_script` 那档动手的方式不同：起的是**本机回路上**一台假 Agent 端（不是往外面连），
    所以不占 `allow_egress` 那道闸；None 时这一格整格不存在，`""`=内置剧本，其余=那份文件。
    """
    items = [
        _probe(
            "runtime",
            "先看 frozen：True 是打包版、False 是源码态，后面每一格的解读都挂在这一格上",
            _runtime,
        ),
        _probe("paths", "读的是 meapet.paths 那一个真值，不是猜的 ~/.meapet", _dirs),
        _probe(
            "config",
            "只打白名单键（值是布尔/枚举/数字，本身不携密）；doctor 不写 config",
            lambda: _config_summary(config),
        ),
        _probe(
            "llm",
            "只覆盖 direct/HTTP 那一路：family 由产品侧 llm_endpoint_family() 归族，api_base 原文不印"
            "（有人把 token 写在查询串里），key 只报在场/缺席与来源；agent 链接那一路的凭据在 "
            "meapet/agent/factory.py 内部，doctor 不复制它，所以本格为缺席不代表整机无凭据",
            lambda: _llm(config),
        ),
        _probe(
            "display_env",
            "环境变量只是线索：XDG_SESSION_TYPE=wayland 而 Qt 用 xcb 时窗口其实在 XWayland 上",
            _display_env,
        ),
        _probe(
            "backend",
            "产品自己算出的后端；只有 wayland 那一档才有 layer-shell 这条路",
            _backend,
        ),
        _probe(
            "layer_shim",
            "文件不在=打包没带上（找 MeaPet.spec）；文件在场但 init 非 0=这台机器的环境或库问题（找 runtime）",
            _layer,
        ),
        _probe(
            "fidus",
            "只查模块规格不 import；不在场=那条分发渠道还没装上 wheel（随包分发是 standing 裁决），不是运行时故障",
            _fidus,
        ),
        _probe(
            "audio",
            "枚举设备要靠它；空表分不出「没声卡」与「pyaudio 没装」，所以这两格分开报",
            _audio_import,
        ),
        _probe(
            "audio_playback",
            "出声那条链的读数：Qt 首选真加载一次（find_spec 查不到被裁掉的 Qt5 库；产品侧那一次 import "
            "在 meapet/desktop/audio.py 的 _qt_media_bundle），回退那三路（startfile/afplay/aplay）"
            "都不吃增益，所以音量档只在 Qt 那一路生效；「底层=加载失败」是**首选不可用**，"
            "不等于没出声——产品那时已经改走回退那一路",
            lambda: _audio_playback(config),
        ),
        _probe(
            "input_devices",
            "pyaudio 在场却枚举到 0 台=这台机器真没有可用输入设备（或格式不通）",
            lambda: _input_devices(config),
        ),
        _probe(
            "asr",
            "两档各问一次是因为产品自己那张表就写着「下了 fp32 却仍读 int8」这一怕；"
            "配置档=缺席 而另一档=在场 ⇒ 不是没下载，是下错了档。模型路径与档位表都取自 meapet.paths 那一个真值",
            lambda: _asr(config),
        ),
        _probe(
            "screens",
            "屏幕几何只有建了 Qt 应用才取得到；platformName=offscreen 时这些数不是真桌面",
            lambda: _screens(with_gui),
        ),
        _probe(
            "input_thresholds",
            "单击/双击那道延迟与分区门票的数：doubleClickInterval 既是**系统给的**、也是人工 2026-10-11 裁的那道阈值"
            "（「比如300ms」是他随口举的例，不是下限）——写死一个小于这一格读数的固定值就拦不住双击；"
            "startDragDistance 若小于产品门票那 35 px，"
            "手抖一下这次点击就被当拖窗吃掉。platformName=offscreen 时读到的是 Qt 默认值，不是这台桌面的设定",
            lambda: _input_thresholds(with_gui),
        ),
        _probe(
            "log_tail",
            "落盘前已过产品侧 redact_text；这里再打一道是防着旧文件里那句没打码的",
            _log_tail,
        ),
    ]
    if skip_layer:
        items = [item for item in items if item.name != "layer_shim"]
    if skip_tts and (measure_speech or measure_repetition):
        # 这两个开关的格子住在 `not skip_tts` 那一块里，只带 `--skip-tts` 时会被整块跳过。
        # 静默吞掉 explicit 请求正是本入口要消灭的那个形状（"点了没反应"），所以它得自己占一行。
        items.append(
            Item(
                name="tts.skipped_on_request",
                value="没跑：--skip-tts 把 --speech-latency / --speech-repetition 要的格一起跳了",
                judge="这一格是**冲突回执**不是读数：两个开关都要真起合成，而 --skip-tts 说的正是"
                      "「一行 TTS 都不要」。想拿这两格的读数就别带 --skip-tts；带着它就是放弃它们",
            )
        )
    if not skip_tts:
        # 两路各探一次：engine 那格在配置里只能是一个值，而两路的坏法完全不同
        # （GSV 坏在解释器与依赖，VITS 坏在权重水化），合起来看只会互相遮蔽。
        for engine in ("gpt_sovits", "vits"):
            items.append(
                _probe(
                    f"tts.{engine}",
                    "前置=不齐 时朗读必然不出声；缺哪一格看事实串（解释器、依赖、权重是否还是 LFS 指针）",
                    lambda eng=engine: _tts(config, eng),
                )
            )
        if measure_speech:
            # 延迟那一档跟在两格前置之后：前置不齐时 `speech_latency` 自己写「未测」，
            # 不会把一发注定失败的读数当成延迟报出去。它排在最后是因为最慢——
            # 前面任何一格炸了都不该先付两分钟的合成钱。
            for engine in ("gpt_sovits", "vits"):
                items.append(
                    _probe(
                        f"tts.latency.{engine}",
                        "本入口唯一的写侧：真起推理子进程并往 tts.output_dir 落 wav（不播放，所以不出声）。"
                        "每发都含模型加载（两引擎都是每句新起进程），第二发暖的只是页缓存；"
                        "秒数必须与同一串里的配置一起读，换个选路/权重/CPU 或 GPU 这数就翻几倍",
                        lambda eng=engine: _tts_latency(config, eng),
                    )
                )
        if measure_repetition:
            # 与延迟同一格位置：它也烧一发真合成（60–135 s），所以同样排在所有便宜格之后。
            for engine in ("gpt_sovits", "vits"):
                items.append(
                    _probe(
                        f"tts.repetition.{engine}",
                        "丁档读数（池项 gsv-chinese-output-token-repetition §6.2）：真合成**一句中文**、"
                        "再用仓内那台 zh-en 流式 zipformer 转写成字面串，报两列——"
                        "「字种 不同/总」与「最长单字连读」。**参照带**：这台随包短参考默认档是 15/5/12，"
                        "补到 ≥6.3 s 的参考后整族掉到 1–2（治复读）；塌句的另一格看**产物秒数**，"
                        "同一句四枚 seed 里有三枚停在 1.02 s。"
                        "**这一格看不见的**：可懂度与「念全没有」——(h) 那发 9/9 零复读而人耳判「后半存在崩坏」，"
                        "① 那发命中两组词而只念出半句。所以它只回答「值不值得去听一次」，"
                        "内容那一档的主判据仍是人工耳朵（口径「只保证VITS的zh正常，其他的只保证出动静」）。"
                        "转写只认 zh/en：日文产物在这台仪器上的「零复读」是仪器失读、不是模型的清白，"
                        "所以这一格钉死中文句、不跟着 tts.voice_lang 走",
                        lambda eng=engine: _tts_repetition(config, eng),
                    )
                )
    if probe_agent:
        items.append(
            _probe(
                "agent_channel",
                "真连一次 WebSocket，**只连回路**（127.0.0.1/::1/localhost）：非回路端点这一格报"
                "「未连接」而不是去连——替这台机器往外发请求不在 doctor 的授权里。连之前先问 TCP："
                "端口没在听就报 `tcp=… 未监听` 且**不发那一发**，这样「这台机器没配 agent 凭据」"
                "与「gateway 挂了」不会同形（产品侧 `hermes.py:596` 把两者都归成「握手失败」，"
                "真因只在 `__cause__` 里，所以那一发跑了之后这里也顺着把真因印出来）。"
                "凭据与地址都走产品侧 factory，doctor 不复制一份；不在场是本案回退链的**预期输入**，"
                "它不等于朗读/点击反馈坏了",
                lambda: _agent_channel(config),
            )
        )
    slots = [("GET", url) for url in probe_sources] + [("POST", url) for url in probe_posts]
    for index, (method, url) in enumerate(slots, start=1):
        items.append(
            _probe(
                f"source.{index}",
                "点击反馈那两个新内容源（关键词检索 / 古诗 API）的**设施取证**：**耗时是端到端实测**，"
                f"不是某个超时参数的转述——两层预算各管一头：每次操作 {SOURCE_PROBE_BUDGET_SECONDS}s"
                f"（连不上/不再吐字节多久放手，不压就吃 `http_async.py:29` 那个 120s 默认）＋ 端到端上限 "
                f"{SOURCE_PROBE_HARD_CAP_SECONDS}s。超判定线 {SOURCE_PROBE_BUDGET_SECONDS}s 的成功照样报成功，"
                "只多挂一句「超预算」——把可用源掐死会把它误读成不可用。"
                "只印状态/字节/首个可见文本片段——200 与一堆字节不说明取回了内容，片段给人过目才说明。"
                "查询串不进读数（有人把 token 写在那儿），头只认写法不印值；POST 的体印出来——延迟几秒"
                "必须连着「用什么查询打出来的」才读得出。"
                "**家族那一格是本案要的**：这台机器 IPv6 出口不稳，auto 那档**连接段**挂的读数只判"
                "「未定」，要 `--probe-family ipv4`/`ipv6` 各跑一遍才敢说这源死活；**收发段**挂的不算"
                "（那是源站慢，不是路不通）；被**端到端上限**掐掉的那一句只说「相位未知」，不替它猜卡在哪；"
                "而**没发出去**那一档（URL 写法/协议被探测自己拒了）连「连上了」三个字都不许出现。"
                "成功那档印实连对端地址。"
                "**默认只连回路**，非回路要显式 --allow-egress，那一步是新增外联、须人工点头；"
                "真站候选清单未定时这条格跑不了，不猜 URL 也不猜端点",
                lambda m=method, u=url: _source_probe(
                    u,
                    allow_egress,
                    method=m,
                    body_json=probe_body,
                    headers=probe_headers,
                    family=probe_family,
                ),
            )
        )
    if agent_script is not None:
        items.append(
            _probe(
                "agent_script",
                "**假端那一趟**：就地起一台只监听 127.0.0.1 的脚本化 Agent Link 服务器，"
                "让**产品自己那个客户端适配器**连上来，按剧本发几发**回合外**的帧，报四件事——"
                "「落队 N/M」= 客户端 `_tool_status_from_payload` 认下来的帧数 / 假端按同一判据"
                "（`state` 与 `safe_text` 都非空）应当让它认的帧数。分母从假端那侧算，所以 "
                "**N=0 而 M=4 是搬运坏了**，**M=0 是这一趟压根没有该入队的帧**（剧本里那些不是"
                "可解析的 `chat.tool_status`，或压根没发到），两种坏在读数里不同形；"
                "「计数 N/M」= 搬回来的那些状态里**至少带出一个可解析计数**的条数 / 假端按客户端"
                "同一个 `safe_counter` 判据应当带出的条数——那是载荷上三个可选键 `call_index`/"
                "`elapsed_ms`/`tokens`（协议文档 §7.2）。桌宠不替 Agent 数、也不把坏值报成 0，所以 "
                "**N=0 而 M>0 是解析坏了**，**M=0 是剧本压根没挂计数**；内置剧本只在头两发挂——"
                "四发全带时这一格会退化成把「落队」再抄一遍。`…40tok` 是**载荷里的 token 计数**，"
                "与末尾「token 长度」那枚握手凭据的**长度**是两回事；"
                "「回执」= 客户端发回来的 `control.error` 的 code，内置剧本里那一发**解析不出且"
                " required** 的帧要的就是 `NO_ACTIVE_TURN`，而**能解析**的那四发按 "
                "`agent_link.py:649-655` 只入队、**不该**有回执——多出回执同样是读数；"
                "「快照」= 客户端发出的 `tools.snapshot` 数，内置剧本 `tools.refresh` 那一步要看它"
                "从 1 变 2；「屏障」= 最后一发 `control.ping` 有没有换来 pong——入站在 `_supervise` "
                "里是**串行** `await _dispatch_frame`，所以 pong 到了才说前三步真的都跑完了，"
                "没到就印「未到齐」，不猜一个数。"
                "**这一格看不见的**：① `take_idle_tool_statuses()` 是这一段能读到而不用撒谎的最深处，"
                "再往后那"
                "一段（`control_bridge.py:189-193` 那个 1 s QTimer → 气泡）无头跑不起来，"
                "所以这一格**不**说明气泡真亮过；② 它不说他配的那条真端点会发这些帧（那是 "
                "`agent_channel` 那一格）；③ 队列上限 16 的**淘汰**行为不测，内置剧本刻意把队列型"
                "帧压在 4 发。握手调 `adapter.probe()`（产品侧那一个函数），凭据走 factory，"
                "但 `MEA_PET_AGENT_LINK_TOKEN` / `MEA_PET_AGENT_TOKEN` 在这一趟里临时摘掉、跑完还回去，"
                "读数只印 token 的**长度**（应当是 0）与帧的 type，不印任何凭据值",
                lambda: _agent_script(config, script_path=agent_script),
            )
        )
    return items


def render_text(items: list[Item], elapsed: float) -> str:
    width = max(len(item.name) for item in items) if items else 0
    lines = [f"meapet --doctor · 只读快照 · 共 {len(items)} 项"]
    lines += [f"{item.name:<{width}}  {item.value}｜判据：{item.judge}" for item in items]
    lines.append(f"总耗时 {elapsed:.2f}s（0=探测跑完，不代表一切正常）")
    return "\n".join(lines)


def _reconfigure_console_utf8() -> None:
    """把本进程的 stdout/stderr 配成 UTF-8（编码错误替换，不炸）。

    取证读数面向「整块贴回来给维护者/AI」，箭头、间隔号这类字符在 GBK 控制台
    （chcp 932/936 的 Windows 默认）打不出 ⇒ 旧形状是 UnicodeEncodeError 直接
    崩掉整个 doctor，一份读数都带不走。errors="replace" 只影响显示保真，不影响
    --doctor-format json 的机器面（json.dumps 本就 ensure_ascii=False + UTF-8）。
    只在真控制台上动；pytest 捕获的 StringIO/NullStream 不动（reconfigure 也没实现）。
    """
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv if argv is not None else sys.argv[1:]))
    _reconfigure_console_utf8()
    from meapet.config.store import load_config

    # 产品侧那些函数里有直接 print 的（`wayland_layer.is_available()` 就是）。
    # doctor 的 stdout 要能整块贴回来或喂进 JSON，所以取证期间的 print 一律串到 stderr。
    started = time.monotonic()
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"[doctor] config 读不出来，取证无从谈起: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    with contextlib.redirect_stdout(sys.stderr):
        items = collect(
            config, with_gui=args.with_gui, skip_tts=args.skip_tts, skip_layer=args.skip_layer,
            measure_speech=args.speech_latency, measure_repetition=args.speech_repetition,
            probe_agent=args.probe_agent,
            probe_sources=tuple(args.probe_source), probe_posts=tuple(args.probe_post),
            probe_body=args.probe_body, probe_headers=tuple(args.probe_header),
            probe_family=args.probe_family, allow_egress=args.allow_egress,
            agent_script=args.agent_script,
        )
    elapsed = time.monotonic() - started
    if args.doctor_format == "json":
        print(json.dumps(
            {"elapsed_s": round(elapsed, 2), "items": [asdict(item) for item in items]},
            ensure_ascii=False,
            indent=2,
        ))
    else:
        print(render_text(items, elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
