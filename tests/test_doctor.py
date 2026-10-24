"""`meapet --doctor` —— 无头只读取证入口的红线。

人工 2026-10-08 放行（逐字见 `~/.Athena/projects/meapet/voice.md`）：「好，可以动码，做无头模式」，
且改判过两条前提：「无头模式是给AI用的」「目的是测试，不论是否打包」。这里钉的是**入口的形状**，
不是各台机器读数的对错：

- **不进 GUI 就不许建 Qt 应用**——无显示服务时那一下是 abort，不是异常，catch 不到（默认关，`--with-gui` 显式开）；
- **只读**——跑一遍不许动 config 文件；
- **探测自身失败要变成一行读数**，不许把 traceback 递给调用方（"无头"两个字判的就是这一格）；
- **stdout 只有报告**——产品侧那些函数里有直接 print 的，取证期间得串到 stderr；
- **两路 TTS 各一条，且吃 diagnostics 那一份真值**——doctor 里重算就是第二套真值；
- **会碰外设的每一档都要显式开关**（`--speech-latency` 起推理落 wav、`--speech-repetition`
  再落一发 wav 另加一次只读 ASR 加载、`--probe-agent` 问 TCP 后
  才开一次 WebSocket 且只连回路）——默认档一格不动、一个包不发。

除 `_port_open` 那条（它要的是真 socket，绑在回环上、不出这台机器）之外，全部遮掉真健康检查
（`MeaTTS.health_check` 在本机 19s）与真连接，要钉的是接线与红线。
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meapet import doctor  # noqa: E402
from meapet.tts.diagnostics import speech_facts, tts_for_engine  # noqa: E402

#: 默认那一臂（`--skip-tts`）的项序，逐格与 `doctor.collect` 对齐；layer 那一格就在 backend 之后。
BASE_ITEMS = [
    "runtime",
    "paths",
    "config",
    "llm",
    "display_env",
    "backend",
    "layer_shim",
    "fidus",
    "audio",
    "audio_playback",
    "input_devices",
    "asr",
    "screens",
    "input_thresholds",
    "log_tail",
]


def _tts_reading_rows(names):
    """`tts.` 前缀里的**读数**格；冲突回执不算（它专门在 skip 时才出现）。"""
    return [n for n in names if n.startswith("tts.") and n != "tts.skipped_on_request"]


class DoctorRequestedTest(unittest.TestCase):
    def test_whole_word_only(self) -> None:
        """`--doctors` 那种参数不该被劫持成取证——劫持了它就是"双击没反应"的另一种形状。"""
        self.assertTrue(doctor.doctor_requested(["python", "pet.py", "--doctor"]))
        self.assertFalse(doctor.doctor_requested(["python", "pet.py", "--doctors"]))
        self.assertFalse(doctor.doctor_requested(["python", "pet.py", "--check"]))
        self.assertFalse(doctor.doctor_requested([]))


class ItemShapeTest(unittest.TestCase):
    def _items(self, **kw):
        return doctor.collect({}, **kw)

    def test_default_has_layer_and_no_tts_when_skipped(self) -> None:
        names = [item.name for item in self._items(skip_tts=True)]
        self.assertEqual(names, BASE_ITEMS)

    def test_skip_layer_drops_the_one_that_actually_inits(self) -> None:
        names = [item.name for item in self._items(skip_tts=True, skip_layer=True)]
        self.assertNotIn("layer_shim", names)
        self.assertEqual(names, [n for n in BASE_ITEMS if n != "layer_shim"])

    def test_both_tts_engines_get_their_own_row(self) -> None:
        """engine 那格在配置里只能是一个值，而两路的坏法完全不同：合起来看只会互相遮蔽。"""
        with mock.patch("meapet.tts.service.MeaTTS.health_check", lambda self: False):
            names = [item.name for item in self._items(skip_tts=False)]
        self.assertIn("tts.gpt_sovits", names)
        self.assertIn("tts.vits", names)

    def test_every_item_carries_its_own_judge(self) -> None:
        """没有判据的读数会被读成"这台机器不行"——本入口要消灭的正是这个形状。"""
        for item in self._items(skip_tts=True):
            self.assertTrue(item.name and item.judge, f"{item.name} 缺判据")


class SpeechLatencyRowTest(unittest.TestCase):
    """`--speech-latency` 是本入口**唯一的写侧**（真起推理子进程 + 落 wav），默认一档都不许跑。

    这一组全部把真健康检查遮成"不齐"：既快又顺带钉住那条顺序——前置不齐时延迟格自己写"未测"，
    不会烧掉一发十几秒到两分钟的合成去换一个失败读数。
    """

    def test_default_run_never_enters_the_synth(self) -> None:
        with mock.patch(
            "meapet.tts.diagnostics.speech_latency", side_effect=AssertionError("默认档不该起合成")
        ):
            names = [item.name for item in doctor.collect({}, skip_tts=False, skip_layer=True)]
        self.assertFalse([n for n in names if n.startswith("tts.latency.")], names)

    def test_the_flag_adds_one_row_per_engine_after_the_prerequisites(self) -> None:
        with mock.patch("meapet.tts.service.MeaTTS.health_check", lambda self: False):
            items = doctor.collect({}, skip_tts=False, skip_layer=True, measure_speech=True)
        self.assertEqual(
            [item.name for item in items if item.name.startswith("tts.")],
            ["tts.gpt_sovits", "tts.vits", "tts.latency.gpt_sovits", "tts.latency.vits"],
            "延迟格排在两格前置之后：前面任何一格炸了都不该先付两分钟的合成钱",
        )
        for item in items:
            if item.name.startswith("tts.latency."):
                self.assertTrue(item.judge, f"{item.name} 缺判据")
                self.assertIn("未测：前置不齐", item.value)

    def test_skip_tts_skips_the_slowest_rows_too(self) -> None:
        """`--skip-tts` 存在的理由就是"那两格最慢"，写侧的延迟格更该跟着跳。"""
        with mock.patch(
            "meapet.tts.diagnostics.speech_latency", side_effect=AssertionError("skip 了就不该起")
        ):
            names = [item.name for item in doctor.collect({}, skip_tts=True, measure_speech=True)]
        self.assertEqual(_tts_reading_rows(names), [], "跳过那一块时只许留冲突回执，不许留读数")

    def test_the_cli_flag_reaches_the_collector(self) -> None:
        self.assertTrue(doctor._parse_args(["--doctor", "--speech-latency"]).speech_latency)
        self.assertFalse(doctor._parse_args(["--doctor"]).speech_latency)
        with mock.patch.object(doctor, "collect", return_value=[]) as collect, mock.patch(
            "meapet.config.store.load_config", return_value={}
        ):
            self.assertEqual(doctor.main(["--doctor", "--speech-latency"]), 0)
        self.assertTrue(collect.call_args.kwargs["measure_speech"])


class SpeechRepetitionRowTest(unittest.TestCase):
    """`--speech-repetition` 那两格：丁档读数，只挂在本入口上（菜单那一侧故意不调，见 diagnostics）。

    钉的四件事与延迟那格同族，另加一条这一族独有的：**仪器起不来时一发合成都不能起**
    （那一发 60–135 s，换不回任何读数）。判据本体在 diagnostics，这里只钉接线。
    """

    def test_default_run_never_enters_the_synth_or_the_asr(self) -> None:
        with mock.patch(
            "meapet.tts.diagnostics.speech_repetition",
            side_effect=AssertionError("默认档不该起合成"),
        ):
            names = [item.name for item in doctor.collect({}, skip_tts=False, skip_layer=True)]
        self.assertFalse([n for n in names if n.startswith("tts.repetition.")], names)

    def test_the_flag_adds_one_row_per_engine_after_the_prerequisites(self) -> None:
        with mock.patch("meapet.tts.service.MeaTTS.health_check", lambda self: False):
            items = doctor.collect(
                {}, skip_tts=False, skip_layer=True, measure_repetition=True
            )
        self.assertEqual(
            [item.name for item in items if item.name.startswith("tts.")],
            [
                "tts.gpt_sovits",
                "tts.vits",
                "tts.repetition.gpt_sovits",
                "tts.repetition.vits",
            ],
            "复读格排在两格前置之后：前面任何一格炸了都不该先付两分钟的合成钱",
        )
        for item in items:
            if item.name.startswith("tts.repetition."):
                self.assertTrue(item.judge, f"{item.name} 缺判据")
                self.assertIn("未测：前置不齐", item.value)

    def test_the_judge_names_both_what_it_reads_and_what_it_cannot(self) -> None:
        """这一格最容易被读成"零复读＝念对了"，判据里必须当场写清它看不见哪一格。"""
        with mock.patch("meapet.tts.service.MeaTTS.health_check", lambda self: False):
            items = doctor.collect(
                {}, skip_tts=False, skip_layer=True, measure_repetition=True
            )
        row = next(i for i in items if i.name == "tts.repetition.gpt_sovits")
        for must in ("字种", "最长单字连读", "15/5/12", "1.02", "人耳", "中文"):
            self.assertIn(must, row.judge, f"{must} 该写在判据里：读数脱了参照带与失读声明就没有读法")

    def test_skip_tts_skips_the_most_expensive_rows_too(self) -> None:
        with mock.patch(
            "meapet.tts.diagnostics.speech_repetition",
            side_effect=AssertionError("skip 了就不该起"),
        ):
            names = [
                item.name for item in doctor.collect({}, skip_tts=True, measure_repetition=True)
            ]
        self.assertEqual(_tts_reading_rows(names), [], "跳过那一块时只许留冲突回执，不许留读数")

    def test_the_cli_flag_reaches_the_collector(self) -> None:
        self.assertTrue(
            doctor._parse_args(["--doctor", "--speech-repetition"]).speech_repetition
        )
        self.assertFalse(doctor._parse_args(["--doctor"]).speech_repetition)
        with mock.patch.object(doctor, "collect", return_value=[]) as collect, mock.patch(
            "meapet.config.store.load_config", return_value={}
        ):
            self.assertEqual(doctor.main(["--doctor", "--speech-repetition"]), 0)
        self.assertTrue(collect.call_args.kwargs["measure_repetition"])

    def test_skip_tts_with_a_speech_flag_answers_instead_of_swallowing_the_request(self) -> None:
        """两格合成住在 `not skip_tts` 里，只带 `--skip-tts` 时会整块跳过。

        把显式请求静默吞掉正是本入口要消灭的形状（"点了没反应"），所以它自己占一行，
        写成**冲突回执**而不是读数。
        """
        for flags in ({"measure_speech": True}, {"measure_repetition": True},
                      {"measure_speech": True, "measure_repetition": True}):
            with self.subTest(**flags):
                items = doctor.collect({}, skip_tts=True, **flags)
                names = [item.name for item in items]
                self.assertIn("tts.skipped_on_request", names)
                row = next(i for i in items if i.name == "tts.skipped_on_request")
                self.assertIn("--skip-tts", row.value)
                self.assertTrue(row.judge, "冲突回执也要有判据：它同样会被读成一条读数")

    def test_plain_skip_tts_still_stays_out_of_the_tts_business(self) -> None:
        """不带那两个开关时，`--skip-tts` 的形状一个字都不能变（那一格是默认排障路径）。"""
        names = [item.name for item in doctor.collect({}, skip_tts=True)]
        self.assertNotIn("tts.skipped_on_request", names)
        self.assertEqual(names, BASE_ITEMS)


class AgentChannelProbeTest(unittest.TestCase):
    """`--probe-agent` 是第二档**会碰外设**的格（先问 TCP，在听才真开一次 WebSocket）。

    这一组要钉的是四件事：默认档一次都不连；**非回路端点报"未连接"而不是去连**（替这台机器
    往外发请求不在 doctor 的授权里，TCP 那一问也算）；端口没在听时**那一发 WebSocket 不发**；
    读数里没有任何凭据形状的东西。
    """

    class _FakeConfig:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

    class _FakeAdapter:
        def __init__(self, base_url: str, probe_error: Exception | None = None) -> None:
            self.config = AgentChannelProbeTest._FakeConfig(base_url)
            self.probe_error = probe_error
            self.probe_calls = 0
            self.close_calls = 0

        async def probe(self):
            self.probe_calls += 1
            if self.probe_error is not None:
                raise self.probe_error
            return types.SimpleNamespace(
                platform="hermes", protocol="json-rpc-2.0", streaming=True, auth_token="不该被印"
            )

        async def close(self):
            self.close_calls += 1

    def _stub_factory(self, adapter, holder=None):
        def factory(config):
            if holder is not None:
                holder.append(config)
            return adapter

        return mock.patch(
            "meapet.agent.factory.create_agent_adapter_from_config", side_effect=factory
        )

    def _port_listening(self):
        """把 TCP 那一问遮成"在听"——这一组要单测的是 WebSocket 那一发，不是端口。"""
        return mock.patch.object(doctor, "_port_open", return_value=True)

    def test_default_run_never_opens_a_socket(self) -> None:
        with mock.patch.object(
            doctor, "_agent_channel", side_effect=AssertionError("默认档不该连")
        ):
            names = [item.name for item in doctor.collect({}, skip_tts=True)]
        self.assertNotIn("agent_channel", names)

    def test_the_flag_adds_the_row_after_the_rest(self) -> None:
        with mock.patch.object(doctor, "_agent_channel", return_value="在场"):
            items = doctor.collect({}, skip_tts=True, probe_agent=True)
        self.assertEqual(items[-1].name, "agent_channel")
        self.assertTrue(items[-1].judge)

    def test_non_loopback_endpoint_is_refused_not_requested(self) -> None:
        """云端/第三方 gateway 地址：这一格必须**不去连**，并把"为什么没连"写在读数里。

        那道 TCP 预问也在禁区里——`getaddrinfo` 加一次 SYN 本身就是往外发的东西，
        所以这里的 `_port_open` 遮成"一调就炸"。
        """
        adapter = self._FakeAdapter("ws://gateway.example.com:9119/api/ws")
        with self._stub_factory(adapter), mock.patch.object(
            doctor, "_port_open", side_effect=AssertionError("非回路端点连 TCP 都不该问")
        ):
            text = doctor._agent_channel({})
        self.assertEqual(adapter.probe_calls, 0, "非回路端点被连了——那就是替这台机器往外发请求")
        self.assertIn("未连接", text)
        self.assertIn("回路", text)

    def test_absent_gateway_is_a_reading_not_a_crash(self) -> None:
        """gateway 没起是本案回退链的**预期输入**，要变成一行读数。"""
        adapter = self._FakeAdapter(
            "ws://127.0.0.1:9119/api/ws", probe_error=ConnectionRefusedError("拒绝连接")
        )
        with self._stub_factory(adapter), self._port_listening():
            text = doctor._agent_channel({})
        self.assertIn("不在场", text)
        self.assertIn("ConnectionRefusedError", text)
        self.assertGreaterEqual(adapter.close_calls, 1)

    def test_present_gateway_prints_capabilities_and_no_credential(self) -> None:
        config = {"llm": {"agent": {"kind": "hermes", "api_key": "sekret-123"}}}
        adapter = self._FakeAdapter("ws://127.0.0.1:9119/api/ws")
        with self._stub_factory(adapter), self._port_listening():
            text = doctor._agent_channel(config)
        self.assertIn("在场", text)
        self.assertIn("platform=hermes", text)
        for leak in ("sekret-123", "不该被印", "auth_token"):
            self.assertNotIn(leak, text, f"凭据形状的东西漏进读数了：{leak}")

    def test_closed_port_never_sends_the_websocket(self) -> None:
        """端口没在听就**不发那一发**。这一道把"这台机器没配 agent 凭据"与"gateway 挂了"
        分开：产品侧 `hermes.py:349` 在无 token 时连接前就抛 `ValueError`，被 `probe()` 末尾
        那个 `except Exception` 归进「握手失败」——两支在读数里同形就没了可指的那一格。
        """
        adapter = self._FakeAdapter("ws://127.0.0.1:9119/api/ws")
        with self._stub_factory(adapter), mock.patch.object(
            doctor, "_port_open", return_value=False
        ):
            text = doctor._agent_channel({})
        self.assertEqual(adapter.probe_calls, 0, "端口关着还发 WebSocket——那一发的预算白付")
        self.assertIn("未监听", text)
        self.assertIn("9119", text)

    def test_the_failure_reason_under_the_wrapper_prints_too(self) -> None:
        """产品侧 `hermes.py:596` 写的是 `raise ValueError("…握手失败") from exc`：外层那句
        对"没配 token"和"对端拒连"完全一样，真因只在 `__cause__` 里。只印外层等于把两支捏回一支。
        """
        outer = ValueError("Hermes WebSocket Gateway 握手失败")
        outer.__cause__ = ValueError("Hermes WebSocket token is required; start hermes serve")
        adapter = self._FakeAdapter("ws://127.0.0.1:9119/api/ws", probe_error=outer)
        with self._stub_factory(adapter), self._port_listening():
            text = doctor._agent_channel({})
        self.assertIn("握手失败", text)
        self.assertIn("token is required", text)

    def test_port_reading_is_taken_from_the_url_not_recomputed(self) -> None:
        self.assertEqual(doctor._endpoint_port("ws://127.0.0.1:9119/api/ws"), 9119)
        self.assertEqual(doctor._endpoint_port("wss://127.0.0.1:8766/x"), 8766)
        self.assertEqual(doctor._endpoint_port("ws://127.0.0.1"), 80)
        self.assertEqual(doctor._endpoint_port("https://127.0.0.1"), 443)
        for bad in ("ws://127.0.0.1:notaport", "ws://127.0.0.1:99999", "不是地址"):
            self.assertIsNone(doctor._endpoint_port(bad), bad)

    def test_probe_does_not_mutate_the_callers_config(self) -> None:
        """工厂会往它拿到的那份 dict 里塞 session_id——所以喂副本，调用方那份不许动。"""
        config = {"llm": {"agent": {"kind": "hermes"}}}
        seen: list[dict] = []

        def factory(mutating_config):
            mutating_config["llm"]["agent"]["session_id"] = "meapet-injected"
            seen.append(mutating_config)
            return self._FakeAdapter("ws://127.0.0.1:9119/api/ws")

        with mock.patch(
            "meapet.agent.factory.create_agent_adapter_from_config", side_effect=factory
        ), self._port_listening():
            doctor._agent_channel(config)
        self.assertEqual(seen[0]["llm"]["agent"]["session_id"], "meapet-injected")
        self.assertEqual(config, {"llm": {"agent": {"kind": "hermes"}}})

    def test_a_broken_config_becomes_a_row_not_a_traceback(self) -> None:
        with mock.patch(
            "meapet.agent.factory.create_agent_adapter_from_config",
            side_effect=ValueError("llm.agent.kind 只能是三种"),
        ):
            text = doctor._agent_channel({"llm": {"agent": {"kind": "nope"}}})
        self.assertIn("适配器构造失败", text)

    def test_loopback_host_truth_table(self) -> None:
        for host in ("127.0.0.1", "::1", "localhost", "127.8.9.10"):
            self.assertTrue(doctor._is_loopback(host), host)
        for host in ("", "gateway.example.com", "10.0.0.5", "169.254.1.1", "8.8.8.8"):
            self.assertFalse(doctor._is_loopback(host), host)

    def test_port_gate_reads_a_real_loopback_listener(self) -> None:
        """`_port_open` 自己也得有一发真的：绑一台回环、听一次、关掉再问一次。

        遮掉它，上面那几条就只剩"探针说在听"；而这道闸的存在理由正是**它比 WebSocket 那句
        更诚实**，所以要拿真 socket 验它两档读数各归各。
        """
        import socket

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        self.assertTrue(doctor._port_open("127.0.0.1", port, budget=1.0))
        listener.close()
        self.assertFalse(doctor._port_open("127.0.0.1", port, budget=1.0))

    def test_the_cli_flag_reaches_the_collector(self) -> None:
        self.assertTrue(doctor._parse_args(["--doctor", "--probe-agent"]).probe_agent)
        self.assertFalse(doctor._parse_args(["--doctor"]).probe_agent)
        with mock.patch.object(doctor, "collect", return_value=[]) as collect, mock.patch(
            "meapet.config.store.load_config", return_value={}
        ):
            self.assertEqual(doctor.main(["--doctor", "--probe-agent"]), 0)
        self.assertTrue(collect.call_args.kwargs["probe_agent"])


class AgentScriptRowTest(unittest.TestCase):
    """`--agent-script` 那一格：就地起假端，量的是**这端客户端那段搬运**。

    真协议那一趟在 `tests/test_fake_agent.py`（真 `AgentLinkAdapter`、真 WebSocket）。这里把
    假端服务器与产品工厂换成桩，只钉 doctor 这侧的五件事：**默认档一台不起**；带开关时那一格
    追加在末尾且不改动前面那些格的形状；判据自己说清它看见到哪为止；跑这一趟时那两枚 token
    env 真被摘掉、跑完原样还回去，读数里只有长度；**「计数」那一格的分子与分母不同源**——
    分子来自客户端搬回来的 `ToolStatus` 字段，分母来自假端那一份引用，所以坏了哪一头都读得出来。
    """

    @staticmethod
    def _result(**overrides):
        """服务器侧那份事实的桩——doctor 只是把它的计数字符化，不参与算。"""
        from meapet.agent.fake_agent import FakeAgentResult

        fields = {
            "sent": 7,
            "received_types": ("control.hello", "control.pong"),
            "error_codes": ("NO_ACTIVE_TURN",),
            "snapshot_seen": 2,
            "barrier_acked": True,
            "replays": 0,
            "queueable_sent": 4,
            "counter_sent": 2,
            "hello_token_lengths": (0,),
        }
        fields.update(overrides)
        return FakeAgentResult(**fields)

    def _patches(self, sink, result, statuses=None):
        """把假端与工厂换成桩，同时把探针期间的环境记进 `sink`。"""
        from meapet.agent.base import ToolStatus

        # 搬回来的那些：默认带上计数，钉的是「客户端给了数就印得出来」这一头。
        idle = (
            [ToolStatus(state="running", safe_text="桩帧", call_index=3, elapsed_ms=1450, tokens=310)]
            if statuses is None
            else statuses
        )

        class _Server:
            url = "ws://127.0.0.1:1/agent-link"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def wait_result(self, _timeout):
                return result

        class _Adapter:
            async def probe(self):
                sink["during"] = {
                    key: os.environ.get(key) for key in doctor.AGENT_SCRIPT_TOKEN_ENVS
                }

            def take_idle_tool_statuses(self):
                return idle

            async def close(self):
                sink["closed"] = True

        return [
            mock.patch(
                "meapet.agent.fake_agent.scripted_agent_server",
                side_effect=lambda script=None: _Server(),
            ),
            mock.patch(
                "meapet.agent.factory.create_agent_adapter_from_config",
                side_effect=lambda config: _Adapter(),
            ),
        ]

    def _run(self, sink, *, config=None, script_path="", result=None, statuses=None):
        with contextlib.ExitStack() as stack:
            for patcher in self._patches(sink, result or self._result(), statuses):
                stack.enter_context(patcher)
            return doctor._agent_script(
                {} if config is None else config, script_path=script_path
            )

    def test_default_run_starts_no_server(self) -> None:
        with mock.patch.object(
            doctor, "_agent_script", side_effect=AssertionError("默认档不该起服务器")
        ):
            names = [item.name for item in doctor.collect({}, skip_tts=True)]
        self.assertNotIn("agent_script", names)
        self.assertEqual(names, BASE_ITEMS)

    def test_the_flag_appends_the_row_after_the_rest(self) -> None:
        with mock.patch.object(doctor, "_agent_script", return_value="在场"):
            items = doctor.collect({}, skip_tts=True, agent_script="")
        names = [item.name for item in items]
        self.assertEqual(names[-1], "agent_script")
        self.assertEqual(names[:-1], BASE_ITEMS, "追加那一格不许改动前面那些格的形状")
        self.assertTrue(items[-1].judge)

    def test_the_judge_says_where_the_reading_stops(self) -> None:
        """判据不许只说"通了"：它得自己讲清最深读到哪、那段看不见的是什么。"""
        with mock.patch.object(doctor, "_agent_script", return_value="在场"):
            judge = doctor.collect({}, skip_tts=True, agent_script="")[-1].judge
        for needle in (
            "take_idle_tool_statuses",       # 最深处
            "control_bridge",                # 再往后那一段（QTimer→气泡）无头跑不起来
            "淘汰",                          # 队列上限 16 那一格不测
            "agent_channel",                 # 它不说他配的那条真端点会发这些帧
            "落队",
            "计数",
            "safe_counter",                # 那一格的分母调的就是客户端那一个函数，不是抄一份
            "未到齐",                        # 屏障没等到就印这句，不猜一个数
            "不印任何凭据",
        ):
            self.assertIn(needle, judge)

    def test_the_token_envs_are_popped_for_the_run_and_restored_after(self) -> None:
        """`factory._resolve_secret` 把这两枚 env 排在**任何** config 字面量之前。

        所以 `auth_token=""` 拦不住它们：不摘，客户端就把这台机器上真那枚 token 递进就地起的
        监听器。这里钉的是摘与还——**跑的时候不在、跑完原样在**，读数里只有长度。
        """
        sentinel = "真那枚 token：它进假端就是往外递"
        previous = {key: os.environ.get(key) for key in doctor.AGENT_SCRIPT_TOKEN_ENVS}
        for key in doctor.AGENT_SCRIPT_TOKEN_ENVS:
            os.environ[key] = sentinel
        sink: dict = {}
        try:
            text = self._run(sink)
            during = sink["during"]
            after = {key: os.environ.get(key) for key in doctor.AGENT_SCRIPT_TOKEN_ENVS}
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertEqual(during, dict.fromkeys(doctor.AGENT_SCRIPT_TOKEN_ENVS))
        self.assertEqual(after, dict.fromkeys(doctor.AGENT_SCRIPT_TOKEN_ENVS, sentinel))
        self.assertNotIn(sentinel, text)
        self.assertIn("token 长度 0", text)

    def test_the_reading_counts_enqueue_from_the_client_not_the_script(self) -> None:
        """分子是 `take_idle_tool_statuses()` 真拿回来的条数，分母是假端按同一判据算的应认数。"""
        text = self._run({})
        self.assertIn("落队 1/4", text)
        self.assertIn("（首帧 running·桩帧）", text)
        self.assertIn("计数 1/2（#3·1450ms·310tok）", text)
        self.assertIn("回执 NO_ACTIVE_TURN", text)
        self.assertIn("快照 2", text)
        self.assertIn("屏障 pong", text)

    def test_the_counter_arm_distinguishes_broken_parse_from_an_empty_script(self) -> None:
        """「计数」那两头的坏法不同形，所以钉成两发：客户端没搬出数 vs 剧本压根没挂。

        第一发分子为 0、分母仍 2 ⇒ **解析/搬运**坏了（假端那侧按同一 `safe_counter` 数过）；
        第二发分子分母都 0 ⇒ 这一趟的剧本里就没有带计数的帧，不是哪里坏了。
        """
        from meapet.agent.base import ToolStatus

        bare = self._run({}, statuses=[ToolStatus(state="running", safe_text="桩帧")])
        self.assertIn("落队 1/4", bare, "计数坏不许顺带把落队那一格也改掉")
        self.assertIn("计数 0/2", bare)
        self.assertNotIn("计数 0/2（", bare, "分子为 0 时没有首帧可印，括号应当整个不出现")

        none_at_all = self._run(
            {},
            statuses=[ToolStatus(state="running", safe_text="桩帧")],
            result=self._result(counter_sent=0),
        )
        self.assertIn("计数 0/0", none_at_all)

    def test_the_counter_note_prints_only_what_arrived(self) -> None:
        """只带了 `elapsed_ms` 那一发：缺的那两项**不印 0**，也不留分隔点。

        印 0 会把「Agent 没给」说成「给了但是零」——那一格正是要人看出差别的地方。
        """
        from meapet.agent.base import ToolStatus

        text = self._run(
            {},
            statuses=[ToolStatus(state="running", safe_text="桩帧", elapsed_ms=501)],
            result=self._result(counter_sent=1),
        )
        self.assertIn("计数 1/1（501ms）", text)
        self.assertNotIn("0tok", text)
        self.assertNotIn("ms·", text)

    def test_a_missing_barrier_prints_that_word_instead_of_a_number(self) -> None:
        text = self._run({}, result=self._result(barrier_acked=False))
        self.assertIn("屏障 未到齐", text)

    def test_a_broken_script_is_a_reading_and_starts_no_server(self) -> None:
        """剧本读不出来 ⇒ 假端压根不起，读数自陈这一步停了。"""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        missing = str(Path(folder.name) / "不存在.json")
        with mock.patch(
            "meapet.agent.fake_agent.scripted_agent_server",
            side_effect=AssertionError("剧本坏了还起了服务器"),
        ):
            text = doctor._agent_script({}, script_path=missing)
        self.assertIn("假端未起", text)
        self.assertIn("不存在.json", text)

    def test_the_cli_flag_reaches_the_collector(self) -> None:
        self.assertIsNone(doctor._parse_args(["--doctor"]).agent_script)
        self.assertEqual(doctor._parse_args(["--doctor", "--agent-script"]).agent_script, "")
        self.assertEqual(
            doctor._parse_args(["--doctor", "--agent-script", "/tmp/x.json"]).agent_script,
            "/tmp/x.json",
        )
        with mock.patch.object(doctor, "collect", return_value=[]) as collect, mock.patch(
            "meapet.config.store.load_config", return_value={}
        ):
            self.assertEqual(doctor.main(["--doctor", "--agent-script"]), 0)
        self.assertEqual(collect.call_args.kwargs["agent_script"], "")


class SourceProbeTest(unittest.TestCase):
    """`--probe-source` 那一档：拿**本机回环上的假 HTTP 服务器**自证四条路径。

    提案 §2 里爬虫与古诗源那两行还闭不了合，卡的是"候选清单未定 + 真跑属新增外联须人工点头"，
    **不是没有工具**。所以这里先把工具钉死——取文 / 两层超时各管一头 / 解析 / 脱敏各一发，
    一个包都不出这台机器（`ThreadingHTTPServer` bind 在 127.0.0.1）。
    """

    @classmethod
    def setUpClass(cls) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            #: HTTP/1.1 + Content-Length 才有 keep-alive——`_peer_note` 取实连对端地址靠的是
            #: 响应还没关连接那一刻（1.0 一答一关，那时 socket 已经没了）。
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # 别把访问日志喷进测试输出
                pass

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                """把**收到的体与头**回显成 JSON：证"那发真按 POST 打出去了"，不证服务器内容。"""
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:  # noqa: BLE001 - 体读不出来就是这一发要过目的东西
                    payload = {}
                echo = {
                    "query": str(payload.get("query", "<体里没有 query>")),
                    "auth": "Bearer 在场" if self.headers.get("Authorization") else "Authorization 缺席",
                    "mode": "Access-Mode 在场" if self.headers.get("X-Tavily-Access-Mode") else "Access-Mode 缺席",
                }
                self._send(200, "application/json", json.dumps(echo, ensure_ascii=False).encode("utf-8"))

            def do_GET(self) -> None:
                if self.path.startswith("/search"):
                    self._send(
                        200,
                        "text/html; charset=utf-8",
                        "<html><head><title>导航</title>"
                        "<script>var hint = '标签里的东西不该出现在可见文本里';</script>"
                        "<body><div class='result'>床前明月光，疑是地上霜</div></body></html>".encode("utf-8"),
                    )
                elif self.path.startswith("/poem"):
                    self._send(
                        200,
                        "application/json",
                        json.dumps(
                            {"data": [{"text": "春眠不觉晓", "author": "孟浩然"}]},
                            ensure_ascii=False,
                        ).encode("utf-8"),
                    )
                elif self.path.startswith("/drip"):
                    #: 一口一口滴：每一读都远不到预算，但**总耗时**过线——真站实测踩过（必应 3.21s、
                    #: 360 4.12s 都 200），所以这一档单独钉住"每次操作"与"端到端"不是一回事。
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", "60")
                    self.end_headers()
                    for index in range(6):
                        self.wfile.write(f"<p>{index}</p>".encode("utf-8").ljust(10, b" "))
                        self.wfile.flush()
                        time.sleep(0.15)
                elif self.path.startswith("/slow"):
                    time.sleep(5)  # 比 3s 的逐请求预算长、比客户端那个 120s 默认短
                    self._send(200, "text/plain", "太晚了，这发不该等到".encode("utf-8"))
                elif self.path.startswith("/boom"):
                    self._send(500, "text/plain", b"upstream broke")
                else:
                    self._send(404, "text/plain", b"nope")

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.server.shutdown)
        cls.addClassCleanup(cls.server.server_close)

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def test_html_gives_visible_text_without_markup_or_script(self) -> None:
        text = doctor._source_probe(self._url("/search"), allow_egress=False)
        self.assertIn("status=200", text)
        self.assertIn("床前明月光", text, "取回了 HTML 却没提出可见文本——那一发等于没证明取到内容")
        self.assertNotIn("标签里的东西不该出现在可见文本里", text)
        for noise in ("<html>", "<div", "class="):
            self.assertNotIn(noise, text, f"markup 漏进读数了：{noise}")

    def test_json_gives_keyed_values_not_a_blob(self) -> None:
        text = doctor._source_probe(self._url("/poem"), allow_egress=False)
        self.assertIn("text=春眠不觉晓", text)
        self.assertIn("author=孟浩然", text)

    def test_per_request_timeout_beats_the_client_default(self) -> None:
        """这一条是本案**逐请求秒级超时**那句约束的真凭据：默认值是 120 s（`http_async.py:29`），
        不显式压它，一个挂站的检索页就能把整份取证摁住两分钟。
        """
        started = time.perf_counter()
        text = doctor._source_probe(self._url("/slow"), allow_egress=False)
        elapsed = time.perf_counter() - started
        self.assertIn("Timeout", text)
        self.assertLess(elapsed, 6.0, f"逐请求预算没生效：真等了 {elapsed:.1f}s")

    def test_trickling_response_is_measured_not_killed(self) -> None:
        """httpx 的 `timeout` 管**每次操作**，不管端到端——滴流响应每一读都不超线，总额却过线。

        真站踩过这一条（必应 3.21 s / 360 4.12 s 都返回 200），所以这里钉两件事：
        ① 超判定线的成功**仍报成功**（掐掉会把可用源误读成不可用）；
        ② 读数里那句「超预算」是真的按端到端耗时判出来的，不是抄超时参数。
        """
        with (
            mock.patch.object(doctor, "SOURCE_PROBE_BUDGET_SECONDS", 0.4),
            mock.patch.object(doctor, "SOURCE_PROBE_HARD_CAP_SECONDS", 30.0),
        ):
            started = time.perf_counter()
            text = doctor._source_probe(self._url("/drip"), allow_egress=False)
            elapsed = time.perf_counter() - started
        self.assertIn("status=200", text, f"滴流被误杀了：{text}")
        self.assertIn("超 0.4s 预算", text)
        self.assertGreater(elapsed, 0.4, "耗时没到判定线，这条测试证不了'超线仍成功'")

    def test_end_to_end_cap_cuts_what_per_read_timeout_cannot(self) -> None:
        """端到端上限是**另一层**：把硬上限压到滴流总时长之下，它必须真把这一发掐掉。

        没有这一层，一个每 2 s 吐一个字节的源可以把取证拖到客户端默认的 120 s。
        """
        with (
            mock.patch.object(doctor, "SOURCE_PROBE_BUDGET_SECONDS", 30.0),
            mock.patch.object(doctor, "SOURCE_PROBE_HARD_CAP_SECONDS", 0.5),
        ):
            started = time.perf_counter()
            text = doctor._source_probe(self._url("/drip"), allow_egress=False)
            elapsed = time.perf_counter() - started
        self.assertIn("TotalTimeout", text, f"端到端上限没生效：{text}")
        self.assertLess(elapsed, 1.5, f"掐太晚（{elapsed:.2f}s），上限没按 0.5s 走")
        #: 这一发是被**我们这层**掐的，不是网络报的——所以它不知道卡在哪一段，读数只许说"相位未知"。
        self.assertIn("相位未知", text, f"端到端那一发要如实说相位不可判：{text}")
        self.assertNotIn("连上了", text, "上限掐掉的那一发不能替它断言连上了")

    def test_non_2xx_and_404_are_readings_not_crashes(self) -> None:
        self.assertIn("status=500", doctor._source_probe(self._url("/boom"), allow_egress=False))
        self.assertIn("status=404", doctor._source_probe(self._url("/gone"), allow_egress=False))

    def test_query_string_never_prints(self) -> None:
        """有人把 token 写在查询串里（`llm` 那格同样的怕）——**整条查询丢掉**，只留 host+path。"""
        text = doctor._source_probe(self._url("/poem?token=sekret-9"), allow_egress=False)
        self.assertIn("/poem", text)
        self.assertNotIn("sekret-9", text)
        self.assertNotIn("token=", text)

    def test_post_echoes_the_body_because_latency_needs_its_config(self) -> None:
        """POST 那一发的**查询体跟着读数印出来**：没有它，"4.06 s" 只是个数，不知道拿什么打出来的。

        这一条同时钉"体真到了服务器"（回显回来的 query）——只印客户端以为自己发了什么，
        等于没证明编码形状对得上（Tavily 收的是 JSON 对象）。
        """
        text = doctor._source_probe(
            self._url("/search"), allow_egress=False, method="POST",
            body_json='{"query":"床前明月光","max_results":3}',
        )
        self.assertIn("POST ", text, f"读数里认不出这是 POST：{text}")
        self.assertIn("status=200", text)
        self.assertIn('体={"query":"床前明月光"', text)
        self.assertIn("query=床前明月光", text, "体没真到服务器：回显里没有那句 query")

    def test_post_without_body_refuses_before_any_packet(self) -> None:
        """没给体就一个包不发：猜一份查询打出来的延迟判不了案，也不许拿它去闭合提案里那一行。"""
        with mock.patch("meapet.http_async.post_json", side_effect=AssertionError("不该发这一个包")):
            text = doctor._source_probe(self._url("/search"), allow_egress=False, method="POST")
        self.assertIn("未请求", text)
        self.assertIn("--probe-body", text)

    def test_post_body_must_be_a_json_object(self) -> None:
        """体写法不对也**不发**：字符串/数组不是这一族 API 的形状，发出去只得到一句 4xx。"""
        for raw in ("不是 JSON", '"just a string"', "[1,2]"):
            text = doctor._source_probe(
                self._url("/search"), allow_egress=False, method="POST", body_json=raw
            )
            self.assertIn("未请求", text, f"这发本该被挡下：{raw} → {text}")

    def test_header_values_never_print_but_do_reach_the_wire(self) -> None:
        """头**只认在场、不印值**（Bearer token 与 keyless 档位都算凭据面），但得证明它真发出去了。"""
        text = doctor._source_probe(
            self._url("/search"), allow_egress=False, method="POST",
            body_json='{"query":"x"}',
            headers=("Authorization: Bearer tvly-SEKRET-9", "X-Tavily-Access-Mode: keyless"),
        )
        self.assertIn("status=200", text)
        self.assertNotIn("tvly-SEKRET-9", text)
        self.assertIn("Bearer 在场", text, "头没到服务器，那一发的档位就是没测过")
        self.assertIn("Access-Mode 在场", text)

    def test_malformed_header_sends_no_packet(self) -> None:
        """`K:V` 写法不对就停下——把「没有冒号的那一枚」整条当名字发出去，是替人猜意图。"""
        with mock.patch("meapet.http_async.get_json", side_effect=AssertionError("不该发这一个包")):
            text = doctor._source_probe(
                self._url("/poem"), allow_egress=False, headers=("没有冒号的那一枚",)
            )
        self.assertIn("未请求", text)

    def test_non_loopback_is_refused_before_any_packet(self) -> None:
        """默认档连 DNS 都不问、一个包不发：`--allow-egress` 是**另一道显式闸**，
        因为真往外发属"新增外联"，按台账口径须人工明确点头。
        """
        with mock.patch(
            "meapet.http_async.get_json", side_effect=AssertionError("不该发这一个包")
        ):
            text = doctor._source_probe("https://gateway.example.com/search", allow_egress=False)
        self.assertIn("未请求", text)
        self.assertIn("--allow-egress", text)

    def test_family_pin_is_a_real_pin_not_a_label(self) -> None:
        """`--probe-family` 必须**真能换地址栈**：钉 ipv6 就该连不上这台只 bind 127.0.0.1 的假服务器。

        印个标签而不换 socket 的话这一格就成了假证据源——本案要问的正是"那条 8 s 无响应到底是
        源站挂了，还是本机 v6 出口挂了"，读数不真就等于没改探针。
        """
        pinned = doctor._source_probe(self._url("/poem"), allow_egress=False, family="ipv4")
        self.assertIn("status=200", pinned)
        self.assertIn("peer=127.0.0.1·ipv4", pinned, f"没印实连对端地址：{pinned}")
        wrong = doctor._source_probe(self._url("/poem"), allow_egress=False, family="ipv6")
        self.assertIn("失败", wrong)
        self.assertNotIn("status=", wrong)
        self.assertIn("家族=ipv6", wrong)

    def test_auto_failure_is_marked_undecided(self) -> None:
        """auto 那档的失败读数必须自带「未定」：双栈由系统挑，把挂写成"这源连不上"就是假证（本机踩过）。"""
        text = doctor._source_probe("http://127.0.0.1:1/poem", allow_egress=False)
        self.assertIn("失败", text)
        self.assertIn("未定", text)
        self.assertIn("--probe-family", text, "没给下一步动作，那句未定等于让人自己想起怎么测")
        pinned = doctor._source_probe("http://127.0.0.1:1/poem", allow_egress=False, family="ipv4")
        self.assertIn("与另一族无关", pinned)
        self.assertNotIn("未定", pinned)

    def test_read_phase_timeout_does_not_blame_the_address_family(self) -> None:
        """挂在**读**那一面就不许劝人分族重测：那一发连上了，只是没在预算内吐完。

        真站 8 发 Tavily 里踩过一发 `ReadTimeout`（3.56s），那时"换一族再试"是把人往错方向支。
        """
        text = doctor._source_probe(self._url("/slow"), allow_egress=False)
        self.assertIn("Timeout", text)
        self.assertIn("与地址栈无关", text)
        self.assertNotIn("--probe-family", text)
        self.assertNotIn("未定", text)

    def test_phase_table_names_each_bucket_once(self) -> None:
        """相位分类是**一张表**，不是就地 `startswith("Connect")` 加一个 `else`。

        那个 `else` 就是上一版的坑：它把"根本没发出去"和"读段慢"读成同一档。
        """
        self.assertEqual(doctor._failure_phase("ConnectTimeout: "), "connect")
        self.assertEqual(doctor._failure_phase("ReadTimeout: x"), "transfer")
        self.assertEqual(doctor._failure_phase("TotalTimeout: 端到端 8.0s 用完"), "capped")
        self.assertEqual(doctor._failure_phase("UnsupportedProtocol: 少协议段"), "not-sent")
        self.assertEqual(doctor._failure_phase("WhoKnowsError: ?"), "unknown")

    def test_a_request_that_never_left_does_not_claim_a_connection(self) -> None:
        """没发出去的那一发，读数里连「连上了」三个字都不许出现。

        2026-10-09 我实测踩到自己这条：shell 里 URL 少写引号被 `&` 截断 → `UnsupportedProtocol`，
        旧那版 `else` 把它归进读段，读数于是印「这一发连上了却没在预算内吐完」——**一个包都没发的
        请求被写成连上了**。这一条钉住那一档只说"探测自己拒的"。
        """
        text = doctor._source_probe("ftp://127.0.0.1:1/x", allow_egress=False)
        self.assertIn("一个包都没发出去", text, text)
        for phrase in ("连上了", "收发", "--probe-family", "未定"):
            self.assertNotIn(phrase, text, f"没发出去的那一发冒充了网络相位：{phrase}")

    def test_collect_numbers_get_then_post_slots_continuously(self) -> None:
        """GET 槽在前、POST 槽接续编号：两份清单合成一条 `source.N` 序列，报告里数得完。"""
        with mock.patch.object(doctor, "_source_probe", return_value="读数"):
            items = doctor.collect(
                {}, skip_tts=True,
                probe_sources=(self._url("/poem"),), probe_posts=(self._url("/search"),),
            )
        self.assertEqual([item.name for item in items[-2:]], ["source.1", "source.2"])

    def test_the_new_probe_flags_reach_the_collector(self) -> None:
        args = doctor._parse_args(
            ["--doctor", "--probe-post", "https://api.tavily.com/search", "--probe-body", '{"query":"x"}',
             "--probe-header", "X-Tavily-Access-Mode: keyless", "--probe-family", "ipv4"]
        )
        self.assertEqual(args.probe_post, ["https://api.tavily.com/search"])
        self.assertEqual(args.probe_body, '{"query":"x"}')
        self.assertEqual(args.probe_header, ["X-Tavily-Access-Mode: keyless"])
        self.assertEqual(args.probe_family, "ipv4")
        self.assertEqual(doctor._parse_args(["--doctor"]).probe_family, "auto")
        self.assertEqual(doctor._parse_args(["--doctor"]).probe_post, [])
        with mock.patch.object(doctor, "collect", return_value=[]) as collect, mock.patch(
            "meapet.config.store.load_config", return_value={}
        ):
            self.assertEqual(
                doctor.main(["--doctor", "--probe-post", "http://127.0.0.1:1/a",
                             "--probe-body", '{"query":"x"}', "--probe-family", "ipv6"]), 0
            )
        self.assertEqual(collect.call_args.kwargs["probe_posts"], ("http://127.0.0.1:1/a",))
        self.assertEqual(collect.call_args.kwargs["probe_family"], "ipv6")
        self.assertEqual(collect.call_args.kwargs["probe_body"], '{"query":"x"}')

    def test_collect_adds_one_row_per_url_and_none_by_default(self) -> None:
        with mock.patch.object(doctor, "_source_probe", return_value="读数"):
            default = [item.name for item in doctor.collect({}, skip_tts=True)]
            self.assertFalse([n for n in default if n.startswith("source.")], default)
            items = doctor.collect(
                {}, skip_tts=True, probe_sources=(self._url("/poem"), self._url("/search"))
            )
        self.assertEqual([item.name for item in items[-2:]], ["source.1", "source.2"])
        for item in items[-2:]:
            self.assertTrue(item.judge)
            self.assertIn("只连回路", item.judge)

    def test_the_cli_flags_reach_the_collector(self) -> None:
        args = doctor._parse_args(
            ["--doctor", "--probe-source", "http://127.0.0.1:1/a", "--probe-source", "http://1/a",
             "--allow-egress"]
        )
        self.assertEqual(args.probe_source, ["http://127.0.0.1:1/a", "http://1/a"])
        self.assertTrue(args.allow_egress)
        self.assertEqual(doctor._parse_args(["--doctor"]).probe_source, [])
        with mock.patch.object(doctor, "collect", return_value=[]) as collect, mock.patch(
            "meapet.config.store.load_config", return_value={}
        ):
            self.assertEqual(
                doctor.main(["--doctor", "--probe-source", "http://127.0.0.1:1/a"]), 0
            )
        self.assertEqual(collect.call_args.kwargs["probe_sources"], ("http://127.0.0.1:1/a",))
        self.assertFalse(collect.call_args.kwargs["allow_egress"])


class ProbeFailureTest(unittest.TestCase):
    def test_a_raising_probe_becomes_a_row_not_a_traceback(self) -> None:
        def boom():
            raise RuntimeError("探测自己炸了")

        item = doctor._probe("boom", "判据", boom)
        self.assertIn("探测自身失败 RuntimeError", item.value)
        self.assertEqual(item.name, "boom")

    def test_secret_in_a_raising_probe_is_redacted(self) -> None:
        item = doctor._probe(
            "boom", "判据", lambda: (_ for _ in ()).throw(RuntimeError("sk-abcdef123456"))
        )
        self.assertNotIn("sk-abcdef123456", item.value)


class QtNotBuiltTest(unittest.TestCase):
    def test_default_collect_does_not_create_a_qt_app(self) -> None:
        """无显示服务时建 Qt 应用是 abort，不是异常——默认那一下必须不发生。"""
        from PyQt5.QtCore import QCoreApplication

        had = QCoreApplication.instance() is not None
        item = doctor._probe("screens", "判据", lambda: doctor._screens(False))
        self.assertEqual(QCoreApplication.instance() is not None, had)
        self.assertIn("未取", item.value)
        self.assertIn("--with-gui", item.value)

    def test_with_gui_uses_the_product_geometry_helper(self) -> None:
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)
        with mock.patch(
            "meapet.desktop.screen_geometry.available_geometry_for", return_value=None
        ) as helper:
            text = doctor._screens(True)
        self.assertIn(f"Qt platformName={app.platformName()}", text)
        helper.assert_called_once_with(None)

    def test_input_thresholds_default_also_builds_nothing(self) -> None:
        """这一格和 screens 同档：默认那一下同样不能建 Qt 应用。"""
        from PyQt5.QtCore import QCoreApplication

        had = QCoreApplication.instance() is not None
        item = doctor._probe("input_thresholds", "判据", lambda: doctor._input_thresholds(False))
        self.assertEqual(QCoreApplication.instance() is not None, had)
        self.assertIn("未取", item.value)

    def test_input_thresholds_reads_the_product_names_not_copied_numbers(self) -> None:
        """产品门票那两个数必须**读自** `live2d_widget` 的名字。

        抄一份数字进来就是第二套真值：改一处漏一处，而这格存在的意义正是把"系统给的门槛"
        和"产品写死的门槛"并排印出来——两边都得是真值才有得比。
        """
        from meapet.desktop import live2d_widget

        with mock.patch.object(doctor, "_qt_app") as app_stub:
            hints = mock.Mock()
            hints.mouseDoubleClickInterval.return_value = 400
            hints.startDragDistance.return_value = 10
            app_stub.return_value = mock.Mock(platformName=lambda: "stub", styleHints=lambda: hints)
            text = doctor._input_thresholds(True)
        self.assertIn("doubleClickInterval=400ms", text)
        self.assertIn("startDragDistance=10px", text)
        self.assertIn(
            f"产品门票={live2d_widget.ZONE_CLICK_MAX_DISTANCE_PX}px"
            f"/{live2d_widget.ZONE_CLICK_MAX_HOLD_SECONDS}s",
            text,
        )

    def test_partition_gate_constants_exist_where_the_widget_reads_them(self) -> None:
        """门票从 `mouseReleaseEvent` 里提出来落到模块名上，条件不能因此变形。"""
        from meapet.desktop import live2d_widget

        self.assertEqual(live2d_widget.ZONE_CLICK_MAX_DISTANCE_PX, 35)
        self.assertEqual(live2d_widget.ZONE_CLICK_MAX_HOLD_SECONDS, 0.4)


class ReadOnlyTest(unittest.TestCase):
    def test_a_full_run_leaves_the_config_file_byte_identical(self) -> None:
        """只读是这个入口唯一的对外承诺：跑一遍取证，不许动任何一个文件。"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            original = json.dumps({"tts": {"enabled": True, "engine": "vits"}})
            path.write_text(original, encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                rc = doctor.main(["--doctor", "--skip-tts", f"--config={path}"])
            self.assertEqual(rc, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_whitelist_itself_has_no_secret_shaped_key(self) -> None:
        """白名单这一层是**结构性**的闸门：将来有人往里加 `llm.api_key`，靠下面那条"报告里没有它"
        是测不出来的——`scrub_secrets` 会先把值清成空串，于是加错键也照样绿。两道分开钉。"""
        secretish = ("api_key", "auth_token", "token", "secret", "password")
        bad = [
            f"{section}.{key}"
            for section, key in doctor.CONFIG_SUMMARY_KEYS
            if any(mark in key.lower() for mark in secretish)
        ]
        self.assertEqual(bad, [], f"这些键名字像凭据，不该进白名单：{bad}")

    def test_llm_row_prints_no_credential_and_no_raw_endpoint(self) -> None:
        """新加的 `llm` 那格是**唯一一处会碰凭据的探测**，所以它的红线单独钉：
        值不印、`api_base` 原文也不印（有人把 token 写在查询串里），只印归族结果与在场布尔。"""
        config = {"llm": {"api_key": "sk-doctor-never-print",
                          "api_base": "https://api.deepseek.com/v1?token=leak-me",
                          "model": "deepseek-chat"}}
        row = doctor._llm(config)
        self.assertNotIn("sk-doctor-never-print", row)
        self.assertNotIn("deepseek.com", row)
        self.assertNotIn("leak-me", row)
        self.assertIn("direct/HTTP key=在场(来源=配置)", row)

    def test_llm_row_states_its_own_scope(self) -> None:
        """这格测不到 agent 链接那一路的凭据——不写明就会把"缺席"读成"这台机器没配任何东西"。"""
        from meapet.doctor import collect

        row = next(item for item in collect({}, skip_tts=True, skip_layer=True)
                   if item.name == "llm")
        self.assertIn("只覆盖 direct/HTTP", row.judge)
        self.assertIn("direct/HTTP", row.value)

    def test_json_format_reports_the_whitelist_only(self) -> None:
        """第二道：即便有人放宽了白名单，`scrub_secrets` 也在打印之前——报告里不该出现凭据值。"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(
                json.dumps({"llm": {"api_key": "sk-doctor-should-not-print"},
                            "tts": {"enabled": True, "engine": "vits", "api_key": "sk-tts-key"}}),
                encoding="utf-8",
            )
            out = io.StringIO()
            err = io.StringIO()
            with mock.patch.object(sys, "stderr", err), redirect_stdout(out):
                rc = doctor.main(["--doctor", "--skip-tts", "--doctor-format", "json",
                                  f"--config={path}"])
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())  # stdout 必须只有 JSON
            self.assertNotIn("sk-doctor-should-not-print", out.getvalue())
            self.assertNotIn("sk-tts-key", out.getvalue())
            self.assertEqual(len(payload["items"]), len(BASE_ITEMS))

    def test_product_prints_are_moved_off_stdout(self) -> None:
        """`wayland_layer.is_available()` 那一类会直接 print；混进报告就没法整块贴回来。"""
        out = io.StringIO()
        with mock.patch("meapet.desktop.wayland_layer.is_available",
                        side_effect=lambda: print("[layer] ✗ 这条不该在 stdout") or False):
            with redirect_stdout(out):
                rc = doctor.main(["--doctor", "--skip-tts"])
        self.assertEqual(rc, 0)
        self.assertNotIn("[layer] ✗", out.getvalue())
        self.assertTrue(out.getvalue().startswith("meapet --doctor"))


class SharedTruthTest(unittest.TestCase):
    def test_doctor_tts_row_uses_the_same_facts_as_the_menu(self) -> None:
        """判据只有一份（`meapet.tts.diagnostics`）；doctor 里重算就开始自证。"""
        config = {"tts": {"enabled": True, "engine": "vits"}}
        with mock.patch("meapet.tts.service.MeaTTS.health_check", lambda self: False):
            row = doctor._tts(config, "vits")
        expected = speech_facts("vits", tts_for_engine(config, "vits"))
        self.assertIn("前置=不齐", row)
        self.assertIn(expected, row)

    def test_doctor_llm_row_carries_the_product_family_verdict(self) -> None:
        """`family=` 必须是 `llm_endpoint_family()` 那个归族结果，不是 doctor 自己认字符串。"""
        from meapet.config.store import llm_endpoint_family

        config = {"llm": {"api_base": "https://api.deepseek.com/v1", "model": "deepseek-chat"}}
        row = doctor._llm(config)
        self.assertIn(llm_endpoint_family(config["llm"]), row)
        self.assertIn("model=deepseek-chat", row)

    def test_doctor_asr_row_asks_both_precisions_of_the_product_table(self) -> None:
        """档位表与目录发现都用 `meapet.paths` 那一个真值；两档分开报是"下错档"那种坏法的判据。"""
        from meapet.paths import VOICE_ASR_MODEL_FILES

        config = {"voice_input": {"enabled": True, "precision": "fp32"}}
        row = doctor._asr(config)
        for precision in VOICE_ASR_MODEL_FILES:
            self.assertIn(precision, row)
        self.assertIn("配置档=fp32", row)
        self.assertIn("fp32(配置档)", row)
        self.assertIn("int8(另一档)", row)


class EntryRoutingTest(unittest.TestCase):
    """`pet.main()` 里 doctor 与 GUI 的先后。

    这一组**必须**先把 `meapet.desktop.app` 换成桩：反向证伪时（把那道拦截改坏）真 GUI
    会在开发者的桌面上开起来并挂住测试进程——一个坏掉的锁应当立刻变红，不该长成一个窗口。
    """

    @staticmethod
    def _gui_stub():
        stub = types.ModuleType("meapet.desktop.app")
        stub.main = lambda: (_ for _ in ()).throw(AssertionError("走到了 GUI 主循环"))
        return stub

    def test_pet_main_routes_to_doctor_before_installing_anything(self) -> None:
        """`--doctor` 拦在 `ensure_pet_dependencies` **之前**：来取证的那台机器往往就是依赖坏的那台，
        而装依赖是写侧动作——先装再问，doctor 就不再是只读入口了。"""
        import pet

        with mock.patch.dict(sys.modules, {"meapet.desktop.app": self._gui_stub()}), \
            mock.patch("meapet.doctor.main", return_value=7) as run, \
            mock.patch("meapet.bootstrap.ensure_pet_dependencies") as install, \
            mock.patch.object(sys, "argv", ["pet.py", "--doctor", "--skip-tts"]):
            rc = pet.main()
        self.assertEqual(rc, 7)
        run.assert_called_once_with(["--doctor", "--skip-tts"])
        install.assert_not_called()

    def test_pet_main_without_the_flag_still_goes_to_the_gui(self) -> None:
        """反向那一半：不带 `--doctor` 时不许被劫持——启动链的形状不能被取证入口改。"""
        import pet

        with mock.patch.dict(sys.modules, {"meapet.desktop.app": self._gui_stub()}), \
            mock.patch("meapet.bootstrap.ensure_pet_dependencies", return_value=False) as install, \
            mock.patch("meapet.doctor.main") as run, \
            mock.patch.object(sys, "argv", ["pet.py", "--check"]):
            rc = pet.main()
        self.assertEqual(rc, 2)
        run.assert_not_called()
        install.assert_called_once()


class DoctorConsoleEncodingTest(unittest.TestCase):
    """GBK 控制台不得把整个 doctor 崩掉（2026-10-24 Windows 冻结包实测的形状）。

    症状：`--doctor` 在 chcp 936 的控制台跑完所有探测后，render/print 阶段
    `UnicodeEncodeError: 'gbk' codec can't encode character '\u21d2'`——
    一份读数都带不走。修复是 `_reconfigure_console_utf8()` 在 main 顶部把
    stdout/stderr 配成 UTF-8+replace。钉两条：有 reconfigure 的真流会被配；
    没有该 API 的流（pytest 的 StringIO 替身等）不炸、不动。
    """

    def test_reconfigure_sets_utf8_on_real_streams(self) -> None:
        import io
        import tempfile

        with tempfile.TemporaryFile(mode="w+b") as raw:
            text_stream = io.TextIOWrapper(raw, encoding="gbk")
            with mock.patch.object(sys, "stdout", text_stream):
                doctor._reconfigure_console_utf8()
                self.assertEqual(
                    (text_stream.encoding or "").lower().replace("-", ""), "utf8"
                )
                # 箭头字符必须写得进去（旧形状在这里 UnicodeEncodeError）
                text_stream.write("\u21d2 \u00b7 ok\n")
                text_stream.flush()
                text_stream.seek(0)
                self.assertIn("\u21d2", text_stream.read())

    def test_reconfigure_tolerates_streams_without_the_api(self) -> None:
        fake = io.StringIO()
        with mock.patch.object(sys, "stdout", fake), mock.patch.object(sys, "stderr", fake):
            doctor._reconfigure_console_utf8()  # 不得抛


if __name__ == "__main__":
    unittest.main()
