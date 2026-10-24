"""VITS「就绪」这句结论只能来自交付脚本自己的 import 面。

出处：Athena 件 meapet-vits-posix-env-path-bug 的 Linux 端到端复跑。空
``vits_env``（有解释器、无包）从前 ``health_check()`` 返回 True——那条分支
把 ``python=True`` 硬写在 checks 里，合取项里也没有它——2.2 s 后合成撞
``ModuleNotFoundError``，用户看到的只是"回退预制语音"。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INFER_SCRIPT = ROOT / "meapet" / "tools" / "vits_infer.py"


def _write_stub(tmp_path: Path, body: str) -> str:
    path = tmp_path / "stub_infer.py"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_probe_says_ok_only_on_the_scripts_own_sentinel(tmp_path):
    from meapet.tts.common import probe_vits_deps

    script = _write_stub(tmp_path, 'print("OK:deps_loaded")\n')
    verdict, detail = probe_vits_deps(sys.executable, script)
    assert verdict == "ok"
    assert "importable" in detail


def test_probe_maps_missing_module_to_missing_with_the_name(tmp_path):
    from meapet.tts.common import probe_vits_deps

    # 模块名要**不存在**，不能挑一个"本机恰好没装"的真包：`.venv` 装了 unidecode
    # 之后，原来那条就变成 rc=0 而判 unknown，测的是本机安装历史而不是探针逻辑。
    script = _write_stub(tmp_path, "import meapet_no_such_dep_xyz  # noqa: F401\n")
    verdict, detail = probe_vits_deps(sys.executable, script)
    assert verdict == "missing"
    # detail 直接进日志/向导文案：得说清缺哪个，不能只回一个布尔
    assert "meapet_no_such_dep_xyz" in detail


def test_probe_detail_picks_the_real_cause_over_an_error_looking_admonition(tmp_path):
    """劝告行里也带 "Error"，取**第一**条会把劝告当结论（本机实测复现过）。

    真现场：``.venv``(无 torch) 跑 ``vits_infer.py --check-deps``，stderr 第 2 行是
    ``⚠ pkg_resources 不可用 (ModuleNotFoundError)；若合成失败请: pip install …``，
    真因 ``ModuleNotFoundError: No module named 'torch'`` 在第 8 行。
    """
    from meapet.tts.common import probe_vits_deps

    script = _write_stub(
        tmp_path,
        "import sys\n"
        "print('  \\u26a0 pkg_resources 不可用 (ModuleNotFoundError)；pip install setuptools', "
        "file=sys.stderr)\n"
        "print('Traceback (most recent call last):', file=sys.stderr)\n"
        "print(\"  File 'vits_infer.py', line 100, in _load_torch_stack\", file=sys.stderr)\n"
        "print(\"ModuleNotFoundError: No module named 'torch'\", file=sys.stderr)\n"
        "sys.exit(1)\n",
    )
    verdict, detail = probe_vits_deps(sys.executable, script)
    assert verdict == "missing"
    assert detail == "ModuleNotFoundError: No module named 'torch'"
    assert "pkg_resources" not in detail


def test_probe_detail_survives_import_errors_without_a_module_name(tmp_path):
    """没有 "No module named" 那一档（DLL 加载失败）仍要出结论，不许退成整段尾巴。"""
    from meapet.tts.common import probe_vits_deps

    script = _write_stub(
        tmp_path,
        "import sys\n"
        "print('SetuptoolsDeprecationWarning: Error prone behaviour', file=sys.stderr)\n"
        "print('ImportError: DLL load failed while importing _C', file=sys.stderr)\n"
        "sys.exit(1)\n",
    )
    verdict, detail = probe_vits_deps(sys.executable, script)
    assert verdict == "missing"
    assert detail == "ImportError: DLL load failed while importing _C"

def test_probe_reports_unknown_for_problems_that_are_not_the_users(tmp_path, monkeypatch):
    """探测自己坏了（超时/脚本不在/没解释器）不许判成"你的环境缺包"。"""
    from meapet.tts import common

    script = _write_stub(tmp_path, "import time; time.sleep(30)\n")
    monkeypatch.setattr(common, "VITS_DEPS_PROBE_TIMEOUT", 1)
    assert common.probe_vits_deps(sys.executable, script)[0] == "unknown"

    # rc 非 0 但不是 ImportError：无法据此判依赖，只能 unknown
    crashy = _write_stub(tmp_path, "import sys; sys.exit(3)\n")
    assert common.probe_vits_deps(sys.executable, crashy)[0] == "unknown"

    assert common.probe_vits_deps("", script)[0] == "unknown"
    assert common.probe_vits_deps(sys.executable, str(tmp_path / "nope.py"))[0] == "unknown"


def test_infer_script_accepts_the_check_deps_flag(tmp_path):
    """--check-deps 必须真被 argparse 接受（rc=2 就是探针与脚本契约断了）。

    用一个只会 ImportError 的假 torch 把 import 掐在第一行：无论本机装没装
    torch，这条测试都是秒级，也不会真的去加载权重。
    """
    fake = tmp_path / "fakepkgs"
    fake.mkdir()
    (fake / "torch.py").write_text("raise ImportError('stub: no torch here')\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake) + os.pathsep + env.get("PYTHONPATH", "")

    # 两种形状都要过：探针现在只发 `--check-deps`，而 Windows 那组实测读数是按带
    # --text/--output 的旧形状取的（rc=0 / 8.67 s），别把那条量过的路改断
    for extra in ([], ["--text", "probe", "--output", os.devnull]):
        proc = subprocess.run(
            [sys.executable, str(INFER_SCRIPT), "--check-deps", *extra],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert proc.returncode != 2, (extra, proc.stderr)
        assert "unrecognized arguments" not in proc.stderr
        assert "required" not in proc.stderr
        assert "ImportError" in proc.stderr


def test_infer_script_still_demands_text_for_synthesis(tmp_path):
    """必填下移 ≠ 文本变可选：不合成才不读它，合成那条路缺 -t 仍是用法错。"""
    proc = subprocess.run(
        [sys.executable, str(INFER_SCRIPT)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2
    assert "--text" in proc.stderr


def test_transient_status_text_has_a_settler():
    """向导写"检测依赖中"这类过渡文案时，必须把状态条交给能结掉它的那条路。

    实测过的坏形状：`_on_vits_env_done` 落了 warning "检测依赖中…"，而探针只写日志
    不碰状态条 ⇒ 状态条永远停在那句过渡文案上（Qt 事件循环里没人会替它改口）。
    """
    src = (ROOT / "wizard" / "page_tts_vits.py").read_text(encoding="utf-8")

    assert "检测依赖中" in src
    assert "status_widget=self.vits_status" in src, (
        "过渡文案没有交割对象：_ensure_vits_deps 需要 status_widget 才会落定结论"
    )


def test_frozen_wizard_reaches_the_pet_exe_dependency_warning():
    """打包版里那句「打包版无法检测 VITS 依赖」得真能出声——从前它是死字。

    `_check_torch` 开头就对 pet exe 返回 False，tier 0️⃣ 于是直接落到下一档，而那条
    警告分支与它用的是同一个谓词 ⇒ 调用图永远走不到 `_ensure_vits_deps` 的 pet-exe
    提示。现在 tier 0️⃣ 判不出来时显式排一次（不 return：后面的档位找到真解释器会把
    结论覆盖掉）。这条钉的是接线本身——谓词与那次调用都得在同一个窗口里。
    """
    lines = (ROOT / "wizard" / "page_tts_vits.py").read_text(
        encoding="utf-8"
    ).splitlines()

    tier0 = [
        i
        for i, ln in enumerate(lines)
        if "_check_torch(_sys.executable)" in ln and not ln.strip().startswith("#")
    ]
    assert len(tier0) == 1, f"tier 0️⃣ 的入口应当唯一，实为 {tier0}"
    guard = next(
        (
            i
            for i in range(tier0[0], min(tier0[0] + 20, len(lines)))
            if "_path_is_pet_exe(_sys.executable)" in lines[i]
        ),
        None,
    )
    assert guard is not None, "tier 0️⃣ 不再判 pet exe ⇒ 那句警告重新变回死字"
    tail = "\n".join(lines[guard : guard + 6])
    assert "_ensure_vits_deps(" in tail, "pet-exe 分支必须显式排一次依赖检测"
    assert "_sys.executable, log" in tail, "排的就是那个 pet exe，别换成别的候选"
    assert "status_widget=self.vits_status" in tail, "状态条也得改口"


def test_probe_and_script_share_one_sentinel_contract():
    """两边各改一半就静默失效：flag 名与哨兵串必须在两处源码里同时出现。"""
    script_src = INFER_SCRIPT.read_text(encoding="utf-8")
    probe_src = (ROOT / "meapet" / "tts" / "common.py").read_text(encoding="utf-8")

    assert '"--check-deps"' in probe_src
    assert "--check-deps" in script_src
    assert "OK:deps_loaded" in script_src
    assert "OK:deps_loaded" in probe_src


def _vits_health_env(tmp_path, monkeypatch):
    """铺一套磁盘事实全齐的 VITS 现场，返回日志行列表（调用方收集读数）。

    磁盘侧准备与健康检查那两条测试同形，共用一份，免得一边改了另一边还绿。
    """
    from meapet.tts import service

    def fake_project_path(*parts):
        return str(tmp_path.joinpath(*parts))

    monkeypatch.setattr(service, "project_path", fake_project_path)
    (tmp_path / "vits_models").mkdir()
    # 真身判据是"不是 LFS 指针"，这里给一段非指针字节即可
    (tmp_path / "vits_models" / "G_latest.pth").write_bytes(b"\x80\x02not-a-pointer")
    (tmp_path / "vits_models" / "finetune_speaker.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vits_core").mkdir()
    (tmp_path / "meapet" / "tools").mkdir(parents=True)
    (tmp_path / "meapet" / "tools" / "vits_infer.py").write_text("", encoding="utf-8")

    lines: list[str] = []
    collector = type("Log", (), {
        "info": staticmethod(lambda msg, *a: lines.append(str(msg) % a if a else str(msg))),
        "warning": staticmethod(lambda msg, *a: lines.append(str(msg) % a if a else str(msg))),
        "error": staticmethod(lambda msg, *a: lines.append(str(msg) % a if a else str(msg))),
        "debug": staticmethod(lambda msg, *a: lines.append(str(msg) % a if a else str(msg))),
    })
    monkeypatch.setattr(service, "log", collector)
    return lines


def test_health_check_vits_stops_claiming_an_unmeasured_python(tmp_path, monkeypatch):
    """子进程分支的日志只报量过的东西，解释器以文件名出现而不是布尔断言。"""
    from meapet.tts import service

    lines = _vits_health_env(tmp_path, monkeypatch)

    tts = service.MeaTTS({
        "tts": {
            "enabled": True,
            "engine": "vits",
            "vits_python": sys.executable,
            # 旋钮走显式配置，不靠 patch service.project_path：模型/配置的默认
            # 取值现在收在 common.vits_model_path() 里，函数内 import 的是
            # meapet.paths，patch service 那一层已经够不着它了（会验到仓库里
            # 那份未水化的 LFS 指针）。
            "vits_model": str(tmp_path / "vits_models" / "G_latest.pth"),
            "vits_config": str(tmp_path / "vits_models" / "finetune_speaker.json"),
        }
    })
    assert tts.health_check() is True
    joined = "\n".join(lines)
    assert "mode=subprocess" in joined
    assert "python=True" not in joined
    assert os.path.basename(sys.executable) in joined


def _inprocess_tts(tmp_path, **extra):
    """显式 ``vits_inprocess: true``、外部解释器留空的 MeaTTS。

    这一格是老代码唯一没验的：进程内那条路的"解释器"就是本进程，而
    health_check 只查 core/model/config 三个磁盘事实，于是宿主 venv 里没有
    torch 也照样报绿，speak() 才撞 ModuleNotFoundError。
    """
    from meapet.tts import service

    cfg = {
        "enabled": True,
        "engine": "vits",
        "vits_inprocess": True,
        "vits_model": str(tmp_path / "vits_models" / "G_latest.pth"),
        "vits_config": str(tmp_path / "vits_models" / "finetune_speaker.json"),
    }
    cfg.update(extra)
    return service.MeaTTS({"tts": cfg})


def _stub_torch_probe(monkeypatch, state: str, detail: str = ""):
    """把 vits_runtime 的后台探针钉成给定结论。

    health_check 只**读**这个结论，所以测试必须钉住它：否则上一个用例留下的
    模块级状态会串到下一个用例里（那正是本判据唯一的谎报面）。
    """
    from meapet.tts.engines import vits_runtime

    calls: list[int] = []

    def fake_probe():
        calls.append(1)
        return state, detail

    monkeypatch.setattr(vits_runtime, "probe_torch_loadable", fake_probe)
    return vits_runtime, calls


def test_inprocess_health_check_is_red_when_this_process_cannot_find_torch(
    tmp_path, monkeypatch
):
    from meapet.tts import service

    lines = _vits_health_env(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "module_present", lambda name: False)
    _stub_torch_probe(monkeypatch, "loaded")

    assert _inprocess_tts(tmp_path).health_check() is False
    joined = "\n".join(lines)
    assert "mode=inprocess" in joined
    assert "torch=False" in joined
    assert "寻不到 torch" in joined


def test_inprocess_health_check_does_not_newly_block_when_torch_is_present(
    tmp_path, monkeypatch
):
    """新判据只往"缺失"方向拦：torch 在，磁盘事实齐就不该因它变红。

    探针处于 probing（后台还没出结论）时**不许**拦——那等于把"我还没量到"
    说成"用户环境坏了"，与 probe_vits_deps 的 unknown 档同一个原则。
    """
    from meapet.tts import service

    lines = _vits_health_env(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "module_present", lambda name: True)
    _stub_torch_probe(monkeypatch, "probing")

    assert _inprocess_tts(tmp_path).health_check() is True
    joined = "\n".join(lines)
    assert "torch=True" in joined
    assert "probe=probing" in joined


def test_health_check_asks_the_probe_once_and_only_when_torch_is_addressable(
    tmp_path, monkeypatch
):
    """探针得真被排上，否则"两级判据"是空话；torch 寻不到时不该白起线程。"""
    from meapet.tts import service

    lines = _vits_health_env(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "module_present", lambda name: True)
    _, calls = _stub_torch_probe(monkeypatch, "probing")

    tts = _inprocess_tts(tmp_path)
    tts.health_check()
    tts.health_check()
    assert len(calls) == 2, "health_check 每次都要把探针问一遍（探针自己保证只跑一次）"

    monkeypatch.setattr(service, "module_present", lambda name: False)
    tts.health_check()
    assert len(calls) == 2, "寻不到 torch 时探针无从谈起，别白起一个 import 线程"


def test_inprocess_health_check_is_red_when_bundled_torch_fails_to_load(
    tmp_path, monkeypatch
):
    """class (b)：`find_spec` 命中、真 import 起不来——这一格只有真 import 能证。

    形状取自 Windows 侧打包版实测回执（Neko_mea）：健康行 torch=True 而
    `import torch` 抛 `WinError 1114 … c10.dll`，speak() 返回 (None, '')。
    改前它报绿；改后探针出结论了就必须报红，且日志要说清是"加载不起来"而不是"寻不到"。
    """
    from meapet.tts import service

    detail = (
        "OSError: Failed to load bundled torch ([WinError 1114] "
        "动态链接库(DLL)初始化例程失败。 Error loading c10.dll)"
    )
    lines = _vits_health_env(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "module_present", lambda name: True)
    _stub_torch_probe(monkeypatch, "failed", detail)

    assert _inprocess_tts(tmp_path).health_check() is False
    joined = "\n".join(lines)
    assert "torch=False" in joined
    assert "probe=failed" in joined
    assert "加载不起来" in joined
    assert "WinError 1114" in joined, "真因得进日志，不能只说'寻不到'"
    assert "寻不到" not in joined, "加载失败报成寻不到会把人指向装 torch，而不是换解释器"


def test_probe_reports_failed_for_an_addressable_but_unloadable_torch(tmp_path):
    """上一用例钉的是结论的消费面；这条在**真子进程**里复现 class (b)。

    临时目录放一个 `torch/__init__.py` 抛 OSError 的假包：`find_spec` 命中（寻得到），
    真 import 失败（加载不起来）。判据不 patch，探针与寻址都跑真的——这样"两级判据"
    不是两个 stub 的排列组合。
    """
    fake = tmp_path / "torch"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        "raise OSError('[WinError 1114] fake c10.dll in a frozen build')\n",
        encoding="utf-8",
    )
    code = (
        "import sys, time\n"
        "from meapet.tts.common import module_present\n"
        "from meapet.tts.engines import vits_runtime as vr\n"
        "addr = module_present('torch')\n"
        "state, detail = vr.probe_torch_loadable()\n"
        "deadline = time.time() + 20\n"
        "while state == vr.PROBE_PROBING and time.time() < deadline:\n"
        "    time.sleep(0.02)\n"
        "    state, detail = vr.torch_probe_status()\n"
        "print('RESULT', addr, state, detail, sep='|')\n"
    )
    env = {
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)]),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    # env 是**刻意**最小化的（否则假 torch 会被真 site-packages 盖掉）。但 Windows 上
    # `Path.home()` 读的是 USERPROFILE，缺它 `import meapet.paths` 直接 RuntimeError
    # "Could not determine home directory"——那是本用例在 Windows 上的红因，与探针无关。
    for _var in ("USERPROFILE", "TEMP", "TMP", "SYSTEMDRIVE", "APPDATA",
                 "LOCALAPPDATA", "USERNAME", "COMSPEC", "PATHEXT"):
        if os.environ.get(_var):
            env[_var] = os.environ[_var]
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")), "")
    assert line, f"子进程没给出读数：{proc.stdout} / {proc.stderr[-400:]}"
    _, addr, state, detail = line.split("|", 3)
    assert addr == "True", "假包必须寻得到——否则测的是 class (a) 不是 (b)"
    assert state == "failed"
    assert "WinError 1114" in detail


def test_module_present_addresses_a_module_without_running_it(tmp_path, monkeypatch):
    """``module_present`` 测的是"寻不寻得到"，不是"跑不跑得起来"。

    这条边界是它敢落在 speak() 路径上的全部理由：find_spec 实测 0.1–0.4 ms，
    真 import torch 3914 ms。第二半断言钉的是"它确实没执行"——否则这个便宜
    判据会偷偷背上加载成本。
    """
    import importlib

    from meapet.tts.common import module_present

    pkg = tmp_path / "addressed_not_executed"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "raise ImportError('executed: __init__ ran')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "addressed_not_executed", raising=False)

    assert module_present("addressed_not_executed") is True
    with pytest.raises(ImportError):
        importlib.import_module("addressed_not_executed")

    assert module_present("definitely_not_installed_here_xyz") is False


def test_health_check_does_not_import_torch_itself():
    """两级判据的重税必须在后台线程里，`health_check` 自己只读结论。

    真 `import torch` 本机实测 3914 ms，而 `health_check()` 落在 `speak()` 路径上；
    一旦有人把它改成同步 import，那条路每次说话都要付这笔钱。这条静态守卫钉的就是
    "service.py 里不许出现装载 torch 的调用"——只数代码行，注释与散文不算。
    """
    banned = ("import torch", "_import_torch(", "_import_runtime(")
    hits = []
    for no, line in enumerate(
        (ROOT / "meapet" / "tts" / "service.py").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if any(b in stripped for b in banned):
            hits.append(f"meapet/tts/service.py:{no}: {stripped}")
    assert hits == [], f"health_check 不能同步装载 torch：{hits}"


def test_speak_path_is_not_taxed_by_the_full_stack_probe():
    """全栈 import 探针 ≈20 s，不许出现在 speak() 的 _ensure_deps 路径上。

    只数调用行、不数散文——上一轮我就把注释里的这个名字扫成了"仍在调用"。
    """

    def call_lines(rel: str) -> list[str]:
        out = []
        for no, line in enumerate(
            (ROOT / rel).read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if "probe_vits_deps(" in stripped and not stripped.startswith("#"):
                out.append(f"{rel}:{no}: {stripped}")
        return out

    assert call_lines("meapet/tts/service.py") == []
    assert call_lines("wizard/page_tts_vits.py"), "就绪判据得有人真的去问"


class TestConcurrentTorchImportSerialized:
    """探针线程与合成线程并发 import torch 必须串行（2026-10-24 冻结包 fastfail 的回归锁）。

    冻结包实测：doctor 的 TTS 前置项排后台探针后，`--speech-latency` 立刻起
    合成，两线程同秒并发走 torch 装载，进程 0xC0000409 原地消失（无任何 Python
    异常）。修复是 `_import_torch` 持 `_LOCK` 串行化。这里用假装载体验证
    接线：主线程占住装载时，第二线程必须被挡在锁外；放行后两边都拿到结论。
    """

    def test_second_thread_waits_for_lock(self) -> None:
        import threading as _th
        from unittest import mock as _mock

        from meapet.tts.engines import vits_runtime as vr

        entered: list[int] = []
        active = _th.Semaphore(0)
        release = _th.Semaphore(0)

        def fake_locked():
            entered.append(1)
            if len(entered) == 1:
                active.release()
                assert release.acquire(timeout=5), "主线程等不到放行"

        with _mock.patch.object(vr, "_import_torch_locked", fake_locked):
            t1 = _th.Thread(target=vr._import_torch)
            t1.start()
            assert active.acquire(timeout=5), "装载线程没进入假件"
            t2 = _th.Thread(target=vr._import_torch)
            t2.start()
            t2.join(timeout=2)
            assert len(entered) == 1, (
                f"第二线程在锁外进了装载体（未串行）：entered={len(entered)}"
            )
            release.release()
            t1.join(timeout=5)
            t2.join(timeout=5)
            assert not (t1.is_alive() or t2.is_alive())
