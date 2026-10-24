"""In-process VITS inference (no external Python subprocess).

Reached when no external interpreter is configured (or the user explicitly
asked for in-process and none is available) — the decision itself lives in
``meapet.tts.common.resolve_vits_route``, never here. When ``tts.vits_python``
does resolve to a real interpreter, the subprocess route is preferred even if
``tts.vits_inprocess`` is true; that override is logged, not silent.

Heavy deps (torch / vits_core) are imported lazily on first synthesis call.

``probe_torch_loadable`` runs that torch import once on a background thread so the
health check can tell "addressable" from "loadable" without paying the tax on the
``speak()`` path — in a frozen build ``find_spec`` hits while ``c10.dll`` can still
fail DllMain, and only a real import distinguishes those two.

This module is also imported by the external ``vits_infer.py`` CLI. Keep its
import graph free of MeaTTS / Qt / heavy desktop code.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Optional

from meapet.paths import project_path
from meapet.log import get_color_logger
from meapet.tts.common import (
    DEFAULT_VITS_CONFIG_NAME,
    DEFAULT_VITS_MODEL_NAME,
    DEFAULT_VITS_SPEAKER,
    resolve_vits_speaker,
)

log = get_color_logger("tts")

_LOCK = threading.RLock()
_MODEL_CACHE: dict[str, Any] = {}
_CORE_READY = False

# ── torch 可加载性探针（一次性、后台）────────────────────────────────────────
# `module_present("torch")` 只证明**寻得到**；打包版里 torch 落在 _MEIPASS 寻得到，
# 而 c10.dll 之类起不来时 import 照样失败——那一格只有真 import 能证。真 import 又不能
# 落在 health_check() 上（它在 speak() 路径里，成功 ~3.9 s / 失败 ~0.8 s），所以这里
# 只在后台跑一次，健康检查读结论：failed 才拦，未出结论一律不拦。
PROBE_UNPROBED = "unprobed"
PROBE_PROBING = "probing"
PROBE_LOADED = "loaded"
PROBE_FAILED = "failed"

_PROBE_LOCK = threading.Lock()
_PROBE_STATE = PROBE_UNPROBED
_PROBE_DETAIL = ""


def torch_probe_status() -> tuple[str, str]:
    """当前的 torch 可加载性结论 ``(state, detail)``，不触发任何加载。"""
    with _PROBE_LOCK:
        return _PROBE_STATE, _PROBE_DETAIL


def _set_probe(state: str, detail: str = "") -> None:
    global _PROBE_STATE, _PROBE_DETAIL
    with _PROBE_LOCK:
        _PROBE_STATE = state
        _PROBE_DETAIL = detail


def _import_torch_locked():
    """真装载 torch（调用方须持 ``_LOCK``），冻结版走 ``torch/lib`` 重试。

    探针与 ``_import_runtime`` **共用**这一条：两边判据不一致的话，探针会对
    "重试其实能救回来"的现场报假红——那正是本判据最不该犯的方向。
    """
    _prepare_torch_dll_search()
    try:
        import torch
    except OSError as exc:
        # Re-try once after forcing torch/lib onto the DLL path using a
        # filesystem probe (import may have failed before torch.__file__).
        meipass = getattr(sys, "_MEIPASS", "") or ""
        torch_lib = os.path.join(meipass, "torch", "lib") if meipass else ""
        if torch_lib and os.path.isdir(torch_lib):
            add_dll = getattr(os, "add_dll_directory", None)
            if callable(add_dll):
                try:
                    add_dll(torch_lib)
                except OSError:
                    pass
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            import torch

            return torch
        raise OSError(
            f"Failed to load bundled torch ({exc}). "
            "On frozen Windows builds, configure tts.vits_python to a real "
            "Python env with torch, or rebuild with a compatible torch wheel."
        ) from exc
    return torch


def _import_torch():
    """把 torch 真加载起来——持 ``_LOCK`` 的唯一入口。

    必须持锁：探针线程与 speak() 主线程可能**同时**走到这里（doctor 的 TTS
    前置项排探针后，``--speech-latency`` 立刻起合成）。CPython 的每模块
    import 锁只保证字节码层互斥；torch 的装载还含原生 DLL 初始化（c10.dll
    DllMain + OpenMP 运行时），Windows 冻结进程里两线程同时初始化同一批 DLL
    的实测形状是 0xC0000409 fastfail（2026-10-24 冻结包实测：探针线程与主
    线程同秒并发 import torch，进程原地消失、无任何 Python 异常）。锁让两个
    入口串行，后到者在 torch 已入 sys.modules 后拿到即时结论。
    """
    with _LOCK:
        return _import_torch_locked()


def _import_torch_for_probe() -> None:
    """探针的判据面：真 import 一次，与合成路径同一条装载逻辑（经持锁入口）。"""
    _import_torch()


def _run_torch_probe() -> None:
    try:
        _import_torch_for_probe()
    except BaseException as exc:  # 后台线程里没人接异常，必须自己落成结论
        _set_probe(PROBE_FAILED, f"{type(exc).__name__}: {exc}")
    else:
        _set_probe(PROBE_LOADED)


def probe_torch_loadable() -> tuple[str, str]:
    """确保后台探针已排上，**立即**返回当前 ``(state, detail)``。

    重复调用只起一个线程：状态从 ``unprobed`` 迁到 ``probing`` 是在锁里做的，
    后到的调用只读结论。
    """
    global _PROBE_STATE
    with _PROBE_LOCK:
        if _PROBE_STATE != PROBE_UNPROBED:
            return _PROBE_STATE, _PROBE_DETAIL
        _PROBE_STATE = PROBE_PROBING
    threading.Thread(
        target=_run_torch_probe, name="vits-torch-probe", daemon=True
    ).start()
    return PROBE_PROBING, "background import in flight"


def _ensure_openjtalk_dict() -> str:
    """Point the bundled dictionary at both names (pyopenjtalk reads only the bare one)."""
    builtin = project_path("dic", "open_jtalk_dic_utf_8-1.11")
    if os.path.isdir(builtin):
        os.environ["MEA_PET_OPEN_JTALK_DICT_DIR"] = builtin
        os.environ["OPEN_JTALK_DICT_DIR"] = builtin
        return builtin
    return os.environ.get("MEA_PET_OPEN_JTALK_DICT_DIR", "")


def _ensure_vits_core_on_path() -> str:
    core = project_path("vits_core")
    if os.path.isdir(core) and core not in sys.path:
        sys.path.insert(0, core)
    return core


def _prepare_torch_dll_search() -> None:
    """Help Windows load torch native DLLs inside a PyInstaller onedir tree.

    WinError 1114 on ``c10.dll`` is commonly caused by the loader not searching
    ``torch/lib`` (and sibling native dirs) when the app is frozen.
    """
    if os.name != "nt":
        return
    candidates: list[str] = []

    # Frozen layout: <_MEIPASS>/torch/lib — do this BEFORE importing torch.
    meipass = getattr(sys, "_MEIPASS", "") or ""
    if meipass:
        candidates.extend(
            [
                os.path.join(meipass, "torch", "lib"),
                os.path.join(meipass, "torch"),
                meipass,
            ]
        )
    # Source / site-packages style
    for entry in list(sys.path):
        if not entry:
            continue
        candidates.append(os.path.join(entry, "torch", "lib"))

    seen: set[str] = set()
    path_prefix: list[str] = []
    for raw in candidates:
        lib_dir = os.path.abspath(raw)
        if not lib_dir or lib_dir in seen or not os.path.isdir(lib_dir):
            continue
        seen.add(lib_dir)
        path_prefix.append(lib_dir)
        add_dll = getattr(os, "add_dll_directory", None)
        if callable(add_dll):
            try:
                add_dll(lib_dir)
            except OSError:
                pass
    if path_prefix:
        os.environ["PATH"] = os.pathsep.join(path_prefix + [os.environ.get("PATH", "")])


def _import_runtime():
    """Lazy-import torch and vits_core modules."""
    global _CORE_READY
    _ensure_openjtalk_dict()
    core = _ensure_vits_core_on_path()
    if not os.path.isdir(core):
        raise FileNotFoundError(f"vits_core not found: {core}")

    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        log.warning(
            "setuptools/pkg_resources missing; VITS text frontend may fail. "
            "Install setuptools==69.5.1 in the build environment."
        )

    try:
        torch = _import_torch()
    except OSError as exc:
        _set_probe(PROBE_FAILED, f"OSError: {exc}")
        raise

    # 真加载成功——把探针结论一并纠正过来（探针可能因为更早的失败停在 failed，
    # 也可能一直没跑过；走到这里说明 torch 在本进程确实加载起来了）。
    _set_probe(PROBE_LOADED)

    import scipy.io.wavfile as wavf
    from torch import LongTensor, no_grad
    import commons
    from text import text_to_sequence
    from models import SynthesizerTrn
    import utils

    _CORE_READY = True
    return {
        "torch": torch,
        "wavf": wavf,
        "LongTensor": LongTensor,
        "no_grad": no_grad,
        "commons": commons,
        "text_to_sequence": text_to_sequence,
        "SynthesizerTrn": SynthesizerTrn,
        "utils": utils,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }


def _get_text(rt: dict, text: str, hps, is_symbol: bool = False):
    text_norm = rt["text_to_sequence"](
        text,
        hps.symbols,
        [] if is_symbol else hps.data.text_cleaners,
    )
    if hps.data.add_blank:
        text_norm = rt["commons"].intersperse(text_norm, 0)
    return rt["LongTensor"](text_norm)


def _load_model(rt: dict, model_path: str, config_path: str):
    hps = rt["utils"].get_hparams_from_file(config_path)
    net_g = rt["SynthesizerTrn"](
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(rt["device"])
    net_g.eval()
    rt["utils"].load_checkpoint(model_path, net_g, None)
    return hps, net_g


def _cache_key(model_path: str, config_path: str) -> str:
    return f"{os.path.abspath(model_path)}::{os.path.abspath(config_path)}"


def get_cached_model(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
):
    """Load (or reuse) the VITS model. Returns ``(hps, net_g, rt)`` or ``(None, None, None)``."""
    model_path = model_path or project_path("vits_models", DEFAULT_VITS_MODEL_NAME)
    config_path = config_path or project_path(
        "vits_models", DEFAULT_VITS_CONFIG_NAME
    )
    if not os.path.isfile(model_path) or not os.path.isfile(config_path):
        return None, None, None

    key = _cache_key(model_path, config_path)
    with _LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        rt = _import_runtime()
        hps, net_g = _load_model(rt, model_path, config_path)
        entry = (hps, net_g, rt)
        _MODEL_CACHE[key] = entry
        return entry


def synthesize_vits(
    text: str,
    output_wav: str,
    *,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    speaker: str = DEFAULT_VITS_SPEAKER,
    noise_scale: float = 0.667,
    noise_scale_w: float = 0.6,
    length_scale: float = 1.0,
) -> str:
    """Synthesize *text* to *output_wav* in-process. Returns the output path."""
    model_path = model_path or project_path("vits_models", DEFAULT_VITS_MODEL_NAME)
    config_path = config_path or project_path(
        "vits_models", DEFAULT_VITS_CONFIG_NAME
    )
    if not text or not str(text).strip():
        raise ValueError("empty VITS text")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"VITS model missing: {model_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"VITS config missing: {config_path}")

    with _LOCK:
        key = _cache_key(model_path, config_path)
        entry = _MODEL_CACHE.get(key)
        if entry is None:
            rt = _import_runtime()
            hps, net_g = _load_model(rt, model_path, config_path)
            entry = (hps, net_g, rt)
            _MODEL_CACHE[key] = entry
        hps, net_g, rt = entry

        speaker_id, speaker_warning = resolve_vits_speaker(
            hps.speakers, speaker
        )
        if speaker_warning:
            log.warning("VITS (in-process): %s", speaker_warning)

        stn_tst = _get_text(rt, text, hps, False)
        device = rt["device"]
        LongTensor = rt["LongTensor"]
        with rt["no_grad"]():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = (
                net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    sid=sid,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )

        parent = os.path.dirname(output_wav) or "."
        os.makedirs(parent, exist_ok=True)
        rt["wavf"].write(output_wav, hps.data.sampling_rate, audio)
        return output_wav


def clear_model_cache() -> None:
    """Drop cached nets (tests / low-memory)."""
    with _LOCK:
        _MODEL_CACHE.clear()
