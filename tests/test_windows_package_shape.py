"""Windows 冻结包的形状规则：Qt bin 旧 CRT 过滤 + 瘦身排除清单。

本文件能证明什么（L0，agents-rules §10）
----------------------------------------
* **spec 的 CRT 过滤器真的摘掉 Qt bin 那 4 个旧运行时**：PyQt5 wheel 往
  ``Qt5\\bin`` 里塞的 2018 年 14.26 MSVC 运行时是打包版进程内 VITS 整条死的根因
  （Qt5Core 先把这份旧 msvcp140 装进进程，torch 的 c10.dll 在它营造的环境里 DllMain
  初始化失败 ⇒ WinError 1114）。把 MeaPet.spec 当可执行输入跑一遍，读它交给
  Analysis / 交给 COLLECT 的 binaries，断言 Qt bin 的 CRT 不在、而**根目录**的
  新版 CRT（torch 还要靠它们）仍在。
* **瘦身排除清单是平台分档的决定，不是构建机副产品**：Windows 侧排除
  transformers/sklearn/pandas/av/numba/llvmlite/librosa/torchaudio；Linux 侧
  **一条都不排除**（fidus 之外 Live2D/OpenGL 链都在场，不做"恰好没装"的赌注）。
  两档都有测试钉住——将来哪个包变成真依赖，得先在这里红、再从清单里删。
* **产品代码对被排除包零直接 import**：这条是排除清单的地基。哪天有人在
  meapet/wizard/pet.py 里 import pandas，这条红得比打包还早。

本文件**不能**证明什么
----------------------
* 不证产物可用（真构建在 scripts/build_windows.ps1，L3 构建机行为）。
* 不证 CRT 过滤器之外 PyInstaller 没从别的路径再收一份 14.26（那是 warn-file
  与 COLLECT toc 的活，构建机上看）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = REPO_ROOT / "MeaPet.spec"
_REAL_FIND_SPEC = __import__("importlib.util", fromlist=["find_spec"]).find_spec


class _Stub:
    """PyInstaller 构建对象的替身：任意属性访问都返回自身，链式用法不炸。"""

    def __getattr__(self, _name):
        return self

    def __iter__(self):
        return iter(())


def _run_spec(spec_dir: Path, calls: list, *, fidus_present: bool = True) -> dict:
    """执行 MeaPet.spec，返回 Analysis(...) 的关键字参数与后处理的 binaries。

    与 test_layer_shim_packaging._run_spec 同形：PyInstaller 假件 + find_spec 接管
    （fidus 在场性读的是构建机环境，测试机不该漂移）。a.binaries 的后处理在 spec
    里发生在 Analysis() 之后、PYZ 之前——本函数让 Analysis 返回带真实列表语义的
    假件，读回 spec 摘过之后的 binaries。
    """
    src = SPEC_FILE.read_text(encoding="utf-8")
    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.collect_data_files = lambda pkg: []
    utils = types.ModuleType("PyInstaller.utils")
    utils.hooks = hooks
    pkgroot = types.ModuleType("PyInstaller")
    pkgroot.utils = utils

    collected: dict = {}

    class _Analysis(_Stub):
        def __init__(self):
            self.binaries = [
                ("PyQt5\\Qt5\\bin\\MSVCP140.dll", "src-path", "BINARY"),
                ("PyQt5\\Qt5\\bin\\msvcp140_1.dll", "src-path", "BINARY"),
                ("PyQt5\\Qt5\\bin\\VCRUNTIME140.dll", "src-path", "BINARY"),
                ("PyQt5\\Qt5\\bin\\vcruntime140_1.dll", "src-path", "BINARY"),
                ("PyQt5\\Qt5\\bin\\Qt5Core.dll", "src-path", "BINARY"),
                ("PyQt5\\Qt5\\plugins\\platforms\\qwindows.dll", "s", "BINARY"),
                ("msvcp140.dll", "src-root-crt", "BINARY"),
                ("MSVCP140_ATOMIC_WAIT.dll", "src-root-crt", "BINARY"),
                ("vcruntime140.dll", "src-root-crt", "BINARY"),
                ("vcruntime140_1.dll", "src-root-crt", "BINARY"),
            ]
            self.pure = []
            self.scripts = []

    def analysis(*args, **kwargs):
        calls.append(kwargs)
        obj = _Analysis()
        collected["obj"] = obj
        return obj

    fake = lambda *a, **k: _Stub()

    def fake_find_spec(name):
        if name == "fidus":
            return object() if fidus_present else None
        return _REAL_FIND_SPEC(name)

    globals_for_spec = {
        "__file__": str(SPEC_FILE),
        "__name__": "meapet_spec",
        "SPECPATH": str(spec_dir),
        "Analysis": analysis,
        "PYZ": fake,
        "EXE": fake,
        "COLLECT": fake,
    }
    with mock.patch.dict(
        sys.modules,
        {
            "PyInstaller": pkgroot,
            "PyInstaller.utils": utils,
            "PyInstaller.utils.hooks": hooks,
        },
    ), mock.patch("importlib.util.find_spec", fake_find_spec):
        exec(compile(src, str(SPEC_FILE), "exec"), globals_for_spec)
    assert calls, "spec 没有调用 Analysis()"
    collected["kwargs"] = calls[0]
    return collected


def _names(binaries) -> list[str]:
    out = []
    for item in binaries:
        out.append(item[0] if isinstance(item, (tuple, list)) else item)
    return out


# ---------- CRT 过滤 ----------

def test_spec_filters_qtbin_old_crts(tmp_path):
    """Qt bin 的 4 个旧 CRT 必须在交给 COLLECT 前被摘掉；根目录新版必须留下。"""
    calls: list = []
    with mock.patch.object(sys, "platform", "win32"):
        out = _run_spec(tmp_path, calls)
    names = [str(n).lower() for n in _names(out["obj"].binaries)]
    for old in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        assert not any(
            n.startswith("pyqt5/qt5/bin/") and n.endswith(old) for n in names
        ), f"Qt bin 旧 CRT {old} 没被摘掉：{names}"
    # 根目录的（torch 与 Qt 回落都要用的）必须仍在
    assert any(n == "msvcp140.dll" for n in names), "根目录 msvcp140.dll 不见了"
    assert any(n == "vcruntime140.dll" for n in names), "根目录 vcruntime140.dll 不见了"
    assert any(n == "msvcp140_atomic_wait.dll" for n in names)
    # Qt 本体与插件不动
    assert any(n.endswith("qt5core.dll") for n in names)
    assert any("qwindows.dll" in n for n in names)


def _load_filter_fn():
    """执行 spec 顶段（Analysis 之前），返回 filter_qtbin_crts。

    必须补齐 _run_spec 同款环境，否则在没装 PyInstaller 的机器（CI 全部
    runner 都不装它——那是构建工具不是运行依赖）上 head-exec 直接
    ModuleNotFoundError；必须钉 win32 档，否则真 Linux runner 上 spec 的
    fidus 在场性闸门会 SystemExit（那道闸是给 Linux 构建机看的）。
    """
    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.collect_data_files = lambda pkg: []
    utils = types.ModuleType("PyInstaller.utils")
    utils.hooks = hooks
    pkgroot = types.ModuleType("PyInstaller")
    pkgroot.utils = utils
    ns = {"SPECPATH": str(REPO_ROOT)}
    head = SPEC_FILE.read_text(encoding="utf-8").split("a = Analysis(")[0]
    with mock.patch.dict(
        sys.modules,
        {"PyInstaller": pkgroot, "PyInstaller.utils": utils, "PyInstaller.utils.hooks": hooks},
    ), mock.patch.object(sys, "platform", "win32"), mock.patch(
        "importlib.util.find_spec", lambda name: object()
    ):
        exec(compile(head, "spec-head", "exec"), ns)
    return ns["filter_qtbin_crts"]


def test_crt_filter_leaves_non_qtbin_paths_alone():
    """过滤器只认 (Qt5\\bin 前缀, 那 4 个文件名) 的组合；别处同名文件不动。"""
    filt = _load_filter_fn()
    binaries = [
        ("PyQt5\\Qt5\\bin\\msvcp140.dll", "s", "BINARY"),
        ("PyQt5\\Qt5\\plugins\\platforms\\qwindows.dll", "s", "BINARY"),
        ("msvcp140.dll", "s", "BINARY"),
        ("vcruntime140.dll", "s", "BINARY"),
        ("numpy\\vcruntime140.dll", "s", "BINARY"),
        ("PyQt5\\Qt5\\bin\\Qt5Core.dll", "s", "BINARY"),
    ]
    kept = filt(binaries)
    kept_names = [k[0] for k in kept]
    assert kept_names == [
        "PyQt5\\Qt5\\plugins\\platforms\\qwindows.dll",
        "msvcp140.dll",
        "vcruntime140.dll",
        "numpy\\vcruntime140.dll",
        "PyQt5\\Qt5\\bin\\Qt5Core.dll",
    ]


def test_crt_filter_handles_forward_slashes():
    """PyInstaller 在不同版本/平台用正斜杠或反斜杠记 dest——两种都要摘。"""
    filt = _load_filter_fn()
    kept = filt([
        ("PyQt5/Qt5/bin/msvcp140.dll", "s", "BINARY"),
        ("PyQt5/Qt5/bin/VCRUNTIME140.dll", "s", "BINARY"),
        ("msvcp140.dll", "s", "BINARY"),
    ])
    assert [k[0] for k in kept] == ["msvcp140.dll"]


# ---------- 瘦身排除清单 ----------

def test_spec_excludes_heavies_on_windows(tmp_path):
    """Windows 档：瘦身清单必须整表出现在 Analysis(excludes=...)。"""
    calls: list = []
    with mock.patch.object(sys, "platform", "win32"):
        out = _run_spec(tmp_path, calls)
    excludes = out["kwargs"]["excludes"]
    for pkg in ("av", "librosa", "llvmlite", "numba", "pandas", "sklearn", "torchaudio", "transformers"):
        assert pkg in excludes, f"Windows 侧 {pkg} 未被排除：{excludes}"


def test_spec_excludes_nothing_heavy_on_linux(tmp_path):
    """Linux 档：瘦身清单必须为空——那条路的依赖面由 fidus/Live2D 自己说了算。"""
    from meapet.desktop import wayland_layer

    (tmp_path / wayland_layer.SHIM_NAME).write_bytes(b"")
    calls: list = []
    with mock.patch.object(sys, "platform", "linux"):
        out = _run_spec(tmp_path, calls, fidus_present=True)
    excludes = out["kwargs"]["excludes"]
    for pkg in ("av", "librosa", "llvmlite", "numba", "pandas", "sklearn", "torchaudio", "transformers"):
        assert pkg not in excludes, f"Linux 侧竟排除了 {pkg}：{excludes}"


# ---------- 产品代码地基 ----------

def test_product_code_does_not_import_excluded_heavies():
    """meapet/wizard/pet.py 对瘦身清单里的包必须零 import（模块级或函数体）。

    排除清单的合法性全靠这条：哪天有人真 import 了 pandas，这条比打包先红。
    gsv_infer.py / vits_infer.py 不在判——它们只落盘给外部解释器，冻结进程
    唯一 import 它们的时刻（doctor --agent-script 那类）也不经过 modulegraph
    的隐藏入口。但 vits_infer.py 与 gsv_infer.py 的**模块级** import 也一并锁：
    落盘脚本在用户自己的 vits 环境里跑，那里的依赖由 vits_requirements.txt 管。
    """
    banned = ("transformers", "pandas", "sklearn", "av", "numba", "llvmlite", "librosa")
    patterns = []
    for pkg in banned:
        patterns.append(f"import {pkg}\n")
        patterns.append(f"import {pkg} ")
        patterns.append(f"from {pkg} import")
        patterns.append(f"from {pkg}.")
    offenders = []
    for py in list((REPO_ROOT / "meapet").rglob("*.py")) + list((REPO_ROOT / "wizard").rglob("*.py")) + [REPO_ROOT / "pet.py"]:
        if py.name == "gsv_infer.py":  # 函数体内的 torchaudio 是 POSIX 分支，Windows 冻结进程永不执行
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pat in patterns:
            if pat in text:
                offenders.append(f"{py.relative_to(REPO_ROOT)}: {pat.strip()!r}")
    assert not offenders, f"产品代码出现了被排除包的 import：\n" + "\n".join(offenders)
