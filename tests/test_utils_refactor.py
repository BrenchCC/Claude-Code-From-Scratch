"""Functional coverage tests for all utils functions with argparse + JSON report."""

import argparse
import ast
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.append(os.getcwd())

import utils.llm_call as llm_call
import utils.reasoning_renderer as reasoning_renderer
import utils.runtime_config as runtime_config
import utils.session_store as session_store
import utils.thinking_policy as thinking_policy
import utils.trace_logger as trace_logger
from utils import __all__ as utils_exports
from utils.llm_call import build_assistant_message, call_chat_completion
from utils.reasoning_renderer import ReasoningRenderer
from utils.runtime_config import RuntimeOptions, runtime_options_from_args
from utils.session_store import SessionStore
from utils.thinking_policy import ThinkingPolicyState, build_thinking_params, resolve_thinking_policy
from utils.trace_logger import TraceLogger


TARGET_MODULES = [
    runtime_config,
    thinking_policy,
    llm_call,
    reasoning_renderer,
    session_store,
    trace_logger,
]

TEST_CONTEXT = {
    "mode": "mock",
    "dotenv_loaded": {},
    "real_queries": [],
    "real_query_logs": [],
    "strict_coverage": True,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI options for this functional test runner.

    Args:
        None.
    """
    parser = argparse.ArgumentParser(
        description = "Run utils function tests and export JSON completion report.",
    )
    parser.add_argument(
        "--mode",
        choices = ["mock", "env", "both"],
        default = "mock",
        help = "mock: virtual IO, env/both: also load .env snapshot for test context.",
    )
    parser.add_argument(
        "--output-json",
        dest = "output_json",
        default = "tests/reports/utils_function_test_report.json",
        help = "Path for JSON report output.",
    )
    parser.add_argument(
        "--pattern",
        default = "",
        help = "Only run tests whose unittest id contains this substring.",
    )
    parser.add_argument(
        "--verbosity",
        type = int,
        default = 2,
        help = "unittest verbosity level.",
    )
    parser.add_argument(
        "--list-functions",
        action = "store_true",
        help = "List discovered utils function targets and exit.",
    )
    parser.add_argument(
        "--list-tests",
        action = "store_true",
        help = "List resolved test IDs (after --pattern filtering) and exit.",
    )
    parser.add_argument(
        "--strict-coverage",
        dest = "strict_coverage",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "When true, uncovered functions make the run fail.",
    )
    parser.add_argument(
        "--load-dotenv",
        dest = "load_dotenv",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Load .env file into process env snapshot.",
    )
    parser.add_argument(
        "--dotenv-path",
        default = ".env",
        help = "Path to .env file.",
    )
    parser.add_argument(
        "--real-query",
        dest = "real_queries",
        action = "append",
        default = [],
        help = "Real prompt text used in env/both mode. Repeatable.",
    )
    return parser.parse_args()


def load_dotenv_file(dotenv_path: Path) -> Dict[str, str]:
    """Load simple KEY=VALUE entries from a dotenv file.

    Args:
        dotenv_path: Path to dotenv file.
    """
    loaded: Dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded

    for raw_line in dotenv_path.read_text(encoding = "utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        env_key = key.strip()
        env_value = raw_value.strip()

        if not env_key:
            continue

        if len(env_value) >= 2 and env_value[0] == env_value[-1] and env_value[0] in {'"', "'"}:
            env_value = env_value[1:-1]

        os.environ.setdefault(env_key, env_value)
        loaded[env_key] = os.environ.get(env_key, env_value)

    return loaded


def discover_function_targets(modules: List[Any]) -> List[str]:
    """Discover module-level functions and class methods from source AST.

    Args:
        modules: Imported python module objects.
    """
    targets: List[str] = []

    for module in modules:
        module_path = Path(module.__file__)
        parsed = ast.parse(module_path.read_text(encoding = "utf-8"))

        for node in parsed.body:
            if isinstance(node, ast.FunctionDef):
                targets.append(f"{module.__name__}.{node.name}")

            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        targets.append(f"{module.__name__}.{node.name}.{child.name}")

    return sorted(targets)


def get_real_queries() -> List[str]:
    """Resolve real query prompts for env-mode integration calls.

    Args:
        None.
    """
    configured = TEST_CONTEXT.get("real_queries") or []
    if configured:
        return list(configured)

    return [
        "请用一句中文回答：你已收到真实环境测试请求。",
        "请给出一条简短学习建议，不超过20字。",
    ]


def build_env_client_and_model() -> Dict[str, Any]:
    """Build OpenAI-compatible client from .env/runtime environment variables.

    Args:
        None.
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()

    if not api_key:
        raise RuntimeError("LLM_API_KEY is required for env real-query mode.")
    if not model:
        raise RuntimeError("LLM_MODEL is required for env real-query mode.")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for env real-query mode.") from exc

    client_kwargs: Dict[str, Any] = {
        "api_key": api_key,
    }
    if base_url:
        client_kwargs["base_url"] = base_url

    return {
        "client": OpenAI(**client_kwargs),
        "model": model,
        "base_url": base_url,
    }


def covers(*targets: str):
    """Attach explicit function coverage targets to one unittest method.

    Args:
        *targets: Fully-qualified function target names.
    """

    def _decorator(test_func: Any) -> Any:
        """Store coverage metadata on the target test method.

        Args:
            test_func: Test method function.
        """
        setattr(test_func, "_covers", list(targets))
        return test_func

    return _decorator


def iter_test_cases(suite: unittest.TestSuite) -> Iterable[unittest.TestCase]:
    """Iterate leaf unittest case instances from a suite tree.

    Args:
        suite: Unittest suite object.
    """
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_test_cases(suite = item)
        else:
            yield item


def extract_covers(test_case: unittest.TestCase) -> List[str]:
    """Read coverage targets from one test case instance.

    Args:
        test_case: Unittest case instance.
    """
    method_name = getattr(test_case, "_testMethodName", None)
    if not method_name:
        return []

    method = getattr(test_case, method_name, None)
    if method is None:
        return []

    return list(getattr(method, "_covers", []))


class RecordingTextResult(unittest.TextTestResult):
    """TextTestResult subclass that stores per-test status for JSON export."""

    def __init__(
        self,
        stream,
        descriptions: bool,
        verbosity: int,
    ):
        """Initialize recording structures.

        Args:
            stream: Output stream used by unittest.
            descriptions: Whether descriptions are enabled.
            verbosity: Verbosity level.
        """
        super().__init__(stream, descriptions, verbosity)
        self.records: Dict[str, Dict[str, Any]] = {}
        self._started_at: Dict[str, float] = {}

    def startTest(self, test: unittest.TestCase) -> None:
        """Record start metadata for one test.

        Args:
            test: Test case instance.
        """
        super().startTest(test)
        test_id = test.id()
        self._started_at[test_id] = time.time()
        self.records[test_id] = {
            "name": test_id,
            "status": "running",
            "message": "",
            "covers": extract_covers(test_case = test),
            "duration_seconds": None,
        }

    def _finish(
        self,
        test: unittest.TestCase,
        status: str,
        message: str = "",
    ) -> None:
        """Finalize one test record.

        Args:
            test: Test case instance.
            status: Final status string.
            message: Optional detail text.
        """
        test_id = test.id()
        started_at = self._started_at.get(test_id, time.time())
        duration = max(0.0, time.time() - started_at)

        record = self.records.setdefault(
            test_id,
            {
                "name": test_id,
                "status": status,
                "message": message,
                "covers": extract_covers(test_case = test),
                "duration_seconds": duration,
            },
        )
        record["status"] = status
        record["message"] = message
        record["duration_seconds"] = round(duration, 6)

    def addSuccess(self, test: unittest.TestCase) -> None:
        """Store success status.

        Args:
            test: Test case instance.
        """
        super().addSuccess(test)
        self._finish(test = test, status = "success")

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        """Store failure status with traceback summary.

        Args:
            test: Test case instance.
            err: Exception tuple.
        """
        super().addFailure(test, err)
        message = self._exc_info_to_string(err, test)
        self._finish(test = test, status = "failure", message = message)

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        """Store error status with traceback summary.

        Args:
            test: Test case instance.
            err: Exception tuple.
        """
        super().addError(test, err)
        message = self._exc_info_to_string(err, test)
        self._finish(test = test, status = "error", message = message)

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        """Store skipped status.

        Args:
            test: Test case instance.
            reason: Skip reason text.
        """
        super().addSkip(test, reason)
        self._finish(test = test, status = "skipped", message = reason)


class _MockCompletions:
    """Mock completion endpoint with captured requests.

    Args:
        handler: Callable that receives request kwargs.
    """

    def __init__(self, handler: Any):
        """Initialize mock completion endpoint.

        Args:
            handler: Request handling callback.
        """
        self._handler = handler
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Capture one create() call and delegate.

        Args:
            **kwargs: Completion request payload.
        """
        self.calls.append(kwargs)
        return self._handler(kwargs)


class _MockClient:
    """Mock client exposing chat.completions.create.

    Args:
        completions: Mock completions endpoint.
    """

    def __init__(self, completions: Any):
        """Initialize mock client wrapper.

        Args:
            completions: Completion endpoint object.
        """
        self.chat = SimpleNamespace(completions = completions)


class _ProbeCompletions:
    """Probe endpoint for thinking capability simulation.

    Args:
        supports_on: Whether enabled probes should pass.
        supports_off: Whether disabled probes should pass.
    """

    def __init__(self, supports_on: bool, supports_off: bool):
        """Initialize probe behavior flags.

        Args:
            supports_on: Pass/fail for enabled params.
            supports_off: Pass/fail for disabled params.
        """
        self.supports_on = supports_on
        self.supports_off = supports_off

    def create(self, **kwargs: Any) -> Any:
        """Simulate probe result according to requested params.

        Args:
            **kwargs: Probe request payload.
        """
        enable_thinking = kwargs.get("enable_thinking")
        reasoning_effort = kwargs.get("reasoning_effort")

        enabled = False
        if enable_thinking is True:
            enabled = True
        if reasoning_effort not in {None, "none"}:
            enabled = True

        if enabled and not self.supports_on:
            raise Exception("unsupported")
        if (not enabled) and not self.supports_off:
            raise Exception("unsupported")

        return SimpleNamespace(ok = True)


class _BrokenModelDump:
    """Object whose model_dump always fails for fallback-path testing."""

    def model_dump(self) -> Any:
        """Raise deterministic runtime error.

        Args:
            None.
        """
        raise RuntimeError("forced model_dump failure")

    def __str__(self) -> str:
        """Return fallback string used by _safe_model_dump.

        Args:
            None.
        """
        return "broken-model-dump"


class _TextLike:
    """Simple helper object exposing text/content style attributes."""

    def __init__(self, text: str = "", content: Any = None, reasoning: Any = None):
        """Initialize helper attributes.

        Args:
            text: Text attribute value.
            content: Content attribute value.
            reasoning: Reasoning attribute value.
        """
        self.text = text
        self.content = content
        self.reasoning = reasoning


class TestRuntimeConfig(unittest.TestCase):
    """Tests for runtime config functions and helpers."""

    @covers("utils.runtime_config.RuntimeOptions.as_dict")
    def test_runtime_options_as_dict(self):
        """Validate RuntimeOptions.as_dict serialization.

        Args:
            self: Test case instance.
        """
        options = RuntimeOptions(
            show_llm_response = True,
            stream = True,
            thinking_mode = "on",
            reasoning_effort = "high",
            reasoning_preview_chars = 8,
            save_session = True,
            session_dir = Path("my_sessions"),
            thinking_capability = "toggle",
            thinking_param_style = "both",
        )

        payload = options.as_dict()

        self.assertTrue(payload["show_llm_response"])
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["thinking_mode"], "on")
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["reasoning_preview_chars"], 8)
        self.assertTrue(payload["save_session"])
        self.assertEqual(payload["session_dir"], "my_sessions")
        self.assertEqual(payload["thinking_capability"], "toggle")
        self.assertEqual(payload["thinking_param_style"], "both")

    @covers("utils.runtime_config.add_runtime_args")
    def test_add_runtime_args_registers_flags(self):
        """Validate argparse flags registration and parsing.

        Args:
            self: Test case instance.
        """
        parser = argparse.ArgumentParser()
        runtime_config.add_runtime_args(parser = parser)

        args = parser.parse_args(
            [
                "--show-llm-response",
                "--stream",
                "--thinking",
                "on",
                "--reasoning-effort",
                "medium",
                "--reasoning-preview-chars",
                "12",
                "--save-session",
                "--session-dir",
                "logs/sessions",
            ]
        )

        self.assertTrue(args.show_llm_response)
        self.assertTrue(args.stream)
        self.assertEqual(args.thinking, "on")
        self.assertEqual(args.reasoning_effort, "medium")
        self.assertEqual(args.reasoning_preview_chars, 12)
        self.assertTrue(args.save_session)
        self.assertEqual(args.session_dir, "logs/sessions")

    @covers(
        "utils.runtime_config.runtime_options_from_args",
        "utils.runtime_config._resolve_bool",
        "utils.runtime_config._resolve_enum",
        "utils.runtime_config._resolve_int",
        "utils.runtime_config._resolve_str",
        "utils.runtime_config._read_normalized_env",
    )
    def test_runtime_options_from_args_and_helpers(self):
        """Validate precedence and all runtime_config private helpers.

        Args:
            self: Test case instance.
        """
        args = Namespace(
            show_llm_response = None,
            stream = None,
            thinking = None,
            reasoning_effort = None,
            reasoning_preview_chars = None,
            save_session = None,
            session_dir = None,
        )

        with patch.dict(
            os.environ,
            {
                "AGENT_SHOW_LLM_RESPONSE": "true",
                "AGENT_STREAM": "off",
                "AGENT_THINKING_MODE": "ON",
                "AGENT_REASONING_EFFORT": "high",
                "AGENT_REASONING_PREVIEW_CHARS": "-20",
                "AGENT_SAVE_SESSION": "yes",
                "AGENT_SESSION_DIR": "   unit_sessions   ",
                "AGENT_THINKING_CAPABILITY": "always",
                "AGENT_THINKING_PARAM_STYLE": "both",
                "X_BOOL": "YeS",
                "X_ENUM": "MEDIUM",
                "X_INT": "33",
                "X_STR": "  value  ",
            },
            clear = True,
        ):
            options = runtime_options_from_args(args = args)

            self.assertTrue(options.show_llm_response)
            self.assertFalse(options.stream)
            self.assertEqual(options.thinking_mode, "on")
            self.assertEqual(options.reasoning_effort, "high")
            self.assertEqual(options.reasoning_preview_chars, 0)
            self.assertTrue(options.save_session)
            self.assertEqual(str(options.session_dir), "unit_sessions")
            self.assertEqual(options.thinking_capability, "always")
            self.assertEqual(options.thinking_param_style, "both")

            resolved_bool = runtime_config._resolve_bool(
                cli_value = None,
                env_name = "X_BOOL",
                default = False,
            )
            resolved_enum = runtime_config._resolve_enum(
                cli_value = None,
                env_name = "X_ENUM",
                default = "none",
                allowed = {"none", "low", "medium", "high"},
            )
            resolved_int = runtime_config._resolve_int(
                cli_value = None,
                env_name = "X_INT",
                default = 0,
            )
            resolved_str = runtime_config._resolve_str(
                cli_value = None,
                env_name = "X_STR",
                default = "fallback",
            )
            normalized_env = runtime_config._read_normalized_env(env_name = "X_ENUM")

        self.assertTrue(resolved_bool)
        self.assertEqual(resolved_enum, "medium")
        self.assertEqual(resolved_int, 33)
        self.assertEqual(resolved_str, "value")
        self.assertEqual(normalized_env, "medium")

        with patch.dict(os.environ, {"X_BOOL": "maybe", "X_INT": "bad"}, clear = True):
            self.assertFalse(
                runtime_config._resolve_bool(
                    cli_value = None,
                    env_name = "X_BOOL",
                    default = False,
                )
            )
            self.assertEqual(
                runtime_config._resolve_int(
                    cli_value = None,
                    env_name = "X_INT",
                    default = 7,
                ),
                7,
            )


class TestThinkingPolicy(unittest.TestCase):
    """Tests for thinking policy resolution and helper functions."""

    @covers(
        "utils.thinking_policy._normalize_setting",
        "utils.thinking_policy._styles_to_probe",
    )
    def test_normalize_setting_and_styles(self):
        """Validate setting normalization and probe style order.

        Args:
            self: Test case instance.
        """
        normalized = thinking_policy._normalize_setting(
            value = " ON ",
            allowed = {"on", "off"},
            default = "off",
        )
        fallback = thinking_policy._normalize_setting(
            value = "invalid",
            allowed = {"on", "off"},
            default = "off",
        )
        auto_styles = thinking_policy._styles_to_probe(param_style = "auto")
        direct_styles = thinking_policy._styles_to_probe(param_style = "both")

        self.assertEqual(normalized, "on")
        self.assertEqual(fallback, "off")
        self.assertEqual(auto_styles, ["enable_thinking", "reasoning_effort", "both"])
        self.assertEqual(direct_styles, ["both"])

    @covers(
        "utils.thinking_policy._resolve_enabled_state",
        "utils.thinking_policy._params_for_enabled_state",
    )
    def test_enabled_state_and_param_translation(self):
        """Validate mode-to-enabled resolution and parameter translation.

        Args:
            self: Test case instance.
        """
        toggle_policy = ThinkingPolicyState(capability = "toggle", param_style = "both")
        always_policy = ThinkingPolicyState(capability = "always", param_style = "enable_thinking")
        never_policy = ThinkingPolicyState(capability = "never", param_style = "reasoning_effort")

        self.assertTrue(thinking_policy._resolve_enabled_state(policy = toggle_policy, mode = "on"))
        self.assertFalse(thinking_policy._resolve_enabled_state(policy = toggle_policy, mode = "off"))
        self.assertTrue(thinking_policy._resolve_enabled_state(policy = always_policy, mode = "auto"))
        self.assertIsNone(thinking_policy._resolve_enabled_state(policy = never_policy, mode = "on"))

        self.assertEqual(
            thinking_policy._params_for_enabled_state(
                style = "enable_thinking",
                enabled = True,
                reasoning_effort = "high",
            ),
            {"enable_thinking": True},
        )
        self.assertEqual(
            thinking_policy._params_for_enabled_state(
                style = "reasoning_effort",
                enabled = False,
                reasoning_effort = "high",
            ),
            {"reasoning_effort": "none"},
        )
        self.assertEqual(
            thinking_policy._params_for_enabled_state(
                style = "both",
                enabled = True,
                reasoning_effort = "low",
            ),
            {
                "enable_thinking": True,
                "reasoning_effort": "low",
            },
        )
        self.assertEqual(
            thinking_policy._params_for_enabled_state(
                style = "unknown",
                enabled = True,
                reasoning_effort = "low",
            ),
            {},
        )

    @covers("utils.thinking_policy._probe_support")
    def test_probe_support(self):
        """Validate probe call success/failure handling.

        Args:
            self: Test case instance.
        """
        supports_client = _MockClient(completions = _ProbeCompletions(supports_on = True, supports_off = True))
        rejects_client = _MockClient(completions = _ProbeCompletions(supports_on = False, supports_off = False))

        self.assertTrue(
            thinking_policy._probe_support(
                client = supports_client,
                model = "m",
                params = {"enable_thinking": True},
            )
        )
        self.assertFalse(
            thinking_policy._probe_support(
                client = rejects_client,
                model = "m",
                params = {"enable_thinking": True},
            )
        )

    @covers("utils.thinking_policy.resolve_thinking_policy")
    def test_resolve_thinking_policy(self):
        """Validate manual and probe-driven policy resolution.

        Args:
            self: Test case instance.
        """
        manual = resolve_thinking_policy(
            client = None,
            model = "",
            capability_setting = "always",
            param_style_setting = "auto",
        )
        auto_none = resolve_thinking_policy(client = None, model = "")

        toggle_client = _MockClient(completions = _ProbeCompletions(supports_on = True, supports_off = True))
        always_client = _MockClient(completions = _ProbeCompletions(supports_on = True, supports_off = False))
        never_client = _MockClient(completions = _ProbeCompletions(supports_on = False, supports_off = True))

        auto_toggle = resolve_thinking_policy(client = toggle_client, model = "m")
        auto_always = resolve_thinking_policy(client = always_client, model = "m")
        auto_never = resolve_thinking_policy(client = never_client, model = "m")

        self.assertEqual(manual.capability, "always")
        self.assertEqual(manual.param_style, "enable_thinking")
        self.assertEqual(auto_none.capability, "never")
        self.assertEqual(auto_none.param_style, "none")
        self.assertEqual(auto_toggle.capability, "toggle")
        self.assertEqual(auto_always.capability, "always")
        self.assertEqual(auto_never.capability, "never")

    @covers("utils.thinking_policy.build_thinking_params")
    def test_build_thinking_params(self):
        """Validate thinking request parameter generation.

        Args:
            self: Test case instance.
        """
        toggle = ThinkingPolicyState(capability = "toggle", param_style = "both")
        always = ThinkingPolicyState(capability = "always", param_style = "enable_thinking")
        never = ThinkingPolicyState(capability = "never", param_style = "none")

        params_on = build_thinking_params(policy = toggle, thinking_mode = "on", reasoning_effort = "high")
        params_off = build_thinking_params(policy = toggle, thinking_mode = "off", reasoning_effort = "high")
        params_auto = build_thinking_params(policy = always, thinking_mode = "auto", reasoning_effort = "medium")
        params_never = build_thinking_params(policy = never, thinking_mode = "on", reasoning_effort = "high")

        self.assertEqual(params_on, {"enable_thinking": True, "reasoning_effort": "high"})
        self.assertEqual(params_off, {"enable_thinking": False, "reasoning_effort": "none"})
        self.assertEqual(params_auto, {"enable_thinking": True})
        self.assertEqual(params_never, {})


class TestReasoningRenderer(unittest.TestCase):
    """Tests for terminal reasoning rendering behavior."""

    @covers(
        "utils.reasoning_renderer.ReasoningRenderer.__init__",
        "utils.reasoning_renderer.ReasoningRenderer.reset_turn",
        "utils.reasoning_renderer.ReasoningRenderer._write",
    )
    def test_renderer_init_reset_write(self):
        """Validate constructor normalization, reset, and write behavior.

        Args:
            self: Test case instance.
        """
        output = io.StringIO()
        renderer = ReasoningRenderer(
            preview_chars = -2,
            output_stream = output,
            input_func = lambda _: "n",
        )

        self.assertEqual(renderer.preview_chars, 0)
        renderer._write(text = "abc")
        self.assertEqual(output.getvalue(), "abc")

        renderer.handle_stream_chunk(chunk = "hello")
        renderer.reset_turn()
        self.assertEqual(renderer._reasoning_buffer, "")
        self.assertEqual(renderer._preview_printed, 0)

    @covers(
        "utils.reasoning_renderer.ReasoningRenderer.handle_stream_chunk",
        "utils.reasoning_renderer.ReasoningRenderer._emit_preview",
        "utils.reasoning_renderer.ReasoningRenderer._emit_fold_notice_if_needed",
    )
    def test_stream_preview_and_fold(self):
        """Validate stream preview accumulation and fold notice behavior.

        Args:
            self: Test case instance.
        """
        output = io.StringIO()
        renderer = ReasoningRenderer(
            preview_chars = 5,
            output_stream = output,
            input_func = lambda _: "n",
        )

        result_a = renderer.handle_stream_chunk(chunk = "abc")
        result_b = renderer.handle_stream_chunk(chunk = "def")

        self.assertEqual(result_a["preview_appended"], "abc")
        self.assertFalse(result_a["folded"])
        self.assertEqual(result_b["preview_appended"], "de")
        self.assertTrue(result_b["folded"])
        self.assertIn("[Reasoning Preview]", output.getvalue())
        self.assertIn("[Reasoning Folded]", output.getvalue())

    @covers(
        "utils.reasoning_renderer.ReasoningRenderer.finalize_turn",
        "utils.reasoning_renderer.ReasoningRenderer._build_hint",
        "utils.reasoning_renderer.ReasoningRenderer._maybe_expand",
    )
    def test_finalize_turn_and_expand_paths(self):
        """Validate finalize hint generation and expand prompt paths.

        Args:
            self: Test case instance.
        """
        output = io.StringIO()
        renderer = ReasoningRenderer(
            preview_chars = 3,
            output_stream = output,
            input_func = lambda _: "r",
        )

        finalized = renderer.finalize_turn(
            full_reasoning = "abcdef",
            stream_mode = False,
            allow_expand_prompt = True,
            interactive = True,
        )

        self.assertTrue(finalized["has_reasoning"])
        self.assertTrue(finalized["folded"])
        self.assertTrue(finalized["expanded"])
        self.assertIn("[Reasoning Available]", output.getvalue())
        self.assertIn("[Reasoning Expanded]", output.getvalue())

        eof_renderer = ReasoningRenderer(
            preview_chars = 3,
            output_stream = io.StringIO(),
            input_func = Mock(side_effect = EOFError()),
        )
        keyboard_renderer = ReasoningRenderer(
            preview_chars = 3,
            output_stream = io.StringIO(),
            input_func = Mock(side_effect = KeyboardInterrupt()),
        )

        self.assertFalse(eof_renderer._maybe_expand(reasoning = "text"))
        self.assertFalse(keyboard_renderer._maybe_expand(reasoning = "text"))
        self.assertEqual(renderer._build_hint(reasoning = "ab", stream_mode = True), "")


class TestSessionStore(unittest.TestCase):
    """Tests for JSONL session persistence behavior."""

    @covers(
        "utils.session_store.SessionStore.__init__",
        "utils.session_store.SessionStore.record_assistant",
        "utils.session_store.SessionStore.record_tool",
        "utils.session_store.SessionStore.get_path",
        "utils.session_store.SessionStore._initialize_output_file",
        "utils.session_store.SessionStore._build_event",
        "utils.session_store.SessionStore._append",
    )
    def test_enabled_store_writes_events(self):
        """Validate enabled store writes meta, assistant, and tool events.

        Args:
            self: Test case instance.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "sessions"
            store = SessionStore(
                enabled = True,
                model = "test/model",
                session_dir = session_dir,
                runtime_options = {"stream": True},
            )

            built = store._build_event(event = "custom", actor = "x")
            self.assertEqual(built["event"], "custom")
            self.assertIn("timestamp", built)

            store.record_assistant(
                actor = "assistant",
                content = "hello",
                reasoning = "r",
                tool_calls = [],
                raw_metadata = {"k": "v"},
            )
            store.record_tool(
                actor = "assistant",
                tool_name = "tool_x",
                arguments = {"a": 1},
                output = {"ok": True},
            )

            path = store.get_path()
            self.assertIsNotNone(path)
            self.assertTrue(path.exists())

            rows = [json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines()]

        self.assertEqual(rows[0]["event"], "meta")
        self.assertEqual(rows[1]["event"], "assistant")
        self.assertEqual(rows[2]["event"], "tool")

    @covers(
        "utils.session_store._sanitize_model_name",
        "utils.session_store._now_iso",
    )
    def test_disabled_store_and_helpers(self):
        """Validate disabled store no-op behavior and helper outputs.

        Args:
            self: Test case instance.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "sessions"
            store = SessionStore(
                enabled = False,
                model = "raw model",
                session_dir = session_dir,
            )

            store.record_assistant(
                actor = "assistant",
                content = "hello",
                reasoning = "",
                tool_calls = [],
            )
            store._append(payload = {"event": "noop"})

            self.assertIsNone(store.get_path())
            self.assertFalse(session_dir.exists())

        sanitized = session_store._sanitize_model_name(model_name = " /abc*model?? ")
        timestamp = session_store._now_iso()

        self.assertEqual(sanitized, "abc_model")
        self.assertRegex(timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


class TestTraceLogger(unittest.TestCase):
    """Tests for trace logger functions and helper summaries."""

    @covers(
        "utils.trace_logger.TraceLogger.__init__",
        "utils.trace_logger.TraceLogger.log_turn",
        "utils.trace_logger._summarize_tool_call",
        "utils.trace_logger._shorten",
    )
    def test_trace_logger_enabled_and_disabled(self):
        """Validate enabled logging, helper formatting, and disabled no-op.

        Args:
            self: Test case instance.
        """
        logger = logging.getLogger("trace_logger_test")
        logger.handlers = []

        active = TraceLogger(enabled = True, logger = logger)

        with self.assertLogs(logger, level = "INFO") as captured:
            active.log_turn(
                actor = "agent",
                assistant_content = "x" * 500,
                tool_calls = [{"function": {"name": "tool_x", "arguments": '{"a":1}'}}],
                assistant_reasoning = "r" * 300,
            )

        combined = "\n".join(captured.output)
        summary = trace_logger._summarize_tool_call(tool_call = {"function": {"name": "sum", "arguments": '{"k":1}'}})
        shortened = trace_logger._shorten(text = "line1\nline2", max_chars = 7)

        self.assertIn("[LLM:agent] assistant:", combined)
        self.assertIn("[LLM:agent] tool_calls:", combined)
        self.assertIn("[LLM:agent] reasoning:", combined)
        self.assertEqual(summary, "sum({\"k\": 1})")
        self.assertEqual(shortened, "line1\\n...")

        mocked_logger = Mock()
        disabled = TraceLogger(enabled = False, logger = mocked_logger)
        disabled.log_turn(actor = "agent", assistant_content = "x", tool_calls = [], assistant_reasoning = "")
        mocked_logger.info.assert_not_called()


class TestLLMCall(unittest.TestCase):
    """Tests for llm_call public and private function behavior."""

    @covers(
        "utils.llm_call._build_request",
        "utils.llm_call._strip_thinking_params",
    )
    def test_build_request_and_strip(self):
        """Validate request assembly and thinking key stripping.

        Args:
            self: Test case instance.
        """
        request = llm_call._build_request(
            model = "m",
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"type": "function"}],
            max_tokens = 55,
            stream = True,
            thinking_params = {"enable_thinking": True, "reasoning_effort": "high"},
        )
        stripped = llm_call._strip_thinking_params(request = request)

        self.assertEqual(request["model"], "m")
        self.assertTrue(request["stream"])
        self.assertIn("enable_thinking", request)
        self.assertIn("reasoning_effort", request)
        self.assertNotIn("enable_thinking", stripped)
        self.assertNotIn("reasoning_effort", stripped)

    @covers(
        "utils.llm_call.call_chat_completion",
        "utils.llm_call.build_assistant_message",
        "utils.llm_call._invoke_with_optional_thinking_retry",
        "utils.llm_call._invoke_once",
        "utils.llm_call._extract_content_from_message",
        "utils.llm_call._extract_reasoning_from_message",
        "utils.llm_call._collect_reasoning_segments",
        "utils.llm_call._normalize_tool_calls",
    )
    def test_non_stream_call_and_assistant_message(self):
        """Validate non-stream result normalization and assistant message creation.

        Args:
            self: Test case instance.
        """
        message = SimpleNamespace(
            content = "hello",
            reasoning = "r1",
            reasoning_content = "r2",
            thinking = "r3",
            tool_calls = [
                SimpleNamespace(
                    id = "call_1",
                    type = "function",
                    function = SimpleNamespace(name = "tool_a", arguments = '{"x":1}'),
                )
            ],
        )
        response = SimpleNamespace(
            choices = [SimpleNamespace(message = message)],
            id = "resp_1",
            model = "model_1",
            usage = SimpleNamespace(model_dump = lambda: {"total_tokens": 1}),
            model_dump = lambda: {"id": "resp_1"},
        )

        completions = _MockCompletions(handler = lambda _: response)
        client = _MockClient(completions = completions)

        result = call_chat_completion(
            client = client,
            model = "model_1",
            messages = [{"role": "user", "content": "hi"}],
        )
        assistant_message = build_assistant_message(result = result)

        self.assertEqual(result.assistant_content, "hello")
        self.assertEqual(result.assistant_reasoning, "r1r2r3")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "tool_a")
        self.assertEqual(result.raw_metadata["stream"], False)
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(assistant_message["content"], "hello")
        self.assertEqual(len(assistant_message["tool_calls"]), 1)

    @covers(
        "utils.llm_call._invoke_stream",
        "utils.llm_call._extract_first_delta",
        "utils.llm_call._extract_content_from_delta",
        "utils.llm_call._extract_reasoning_from_delta",
        "utils.llm_call._merge_stream_tool_calls",
    )
    def test_stream_call_and_callbacks(self):
        """Validate stream chunk merge for content, reasoning, and tools.

        Args:
            self: Test case instance.
        """
        chunk_empty = SimpleNamespace(choices = [])
        chunk_1 = SimpleNamespace(
            id = "chunk_1",
            model = "model_1",
            choices = [
                SimpleNamespace(
                    delta = SimpleNamespace(
                        content = "Hel",
                        reasoning = "R1",
                        tool_calls = [{"index": 0, "id": "call_1", "function": {"name": "fo", "arguments": '{"x"'}}],
                    )
                )
            ],
        )
        chunk_2 = SimpleNamespace(
            id = "chunk_2",
            model = "model_1",
            choices = [
                SimpleNamespace(
                    delta = SimpleNamespace(
                        content = [
                            {"type": "text", "text": "lo"},
                            {"type": "reasoning", "text": "hidden"},
                        ],
                        thinking = "R2",
                        tool_calls = [{"index": 0, "function": {"name": "o", "arguments": ':1}'}}],
                    )
                )
            ],
        )

        completions = _MockCompletions(handler = lambda _: [chunk_empty, chunk_1, chunk_2])
        client = _MockClient(completions = completions)

        content_chunks: List[str] = []
        reasoning_chunks: List[str] = []

        result = call_chat_completion(
            client = client,
            model = "model_1",
            messages = [{"role": "user", "content": "hi"}],
            stream = True,
            on_content_chunk = content_chunks.append,
            on_reasoning_chunk = reasoning_chunks.append,
        )

        self.assertEqual(result.assistant_content, "Hello")
        self.assertEqual(result.assistant_reasoning, "R1R2hidden")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "fo")
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], '{"x":1}')
        self.assertEqual(content_chunks, ["Hel", "lo"])
        self.assertEqual(reasoning_chunks, ["R1", "R2hidden"])
        self.assertEqual(result.raw_metadata["stream"], True)
        self.assertEqual(result.raw_metadata["chunk_count"], 3)

    @covers(
        "utils.llm_call._looks_like_thinking_param_error",
    )
    def test_thinking_error_signal_detection(self):
        """Validate thinking-param related error detection heuristic.

        Args:
            self: Test case instance.
        """
        self.assertTrue(llm_call._looks_like_thinking_param_error(Exception("unknown parameter: enable_thinking")))
        self.assertTrue(llm_call._looks_like_thinking_param_error(Exception("reasoning_effort is not permitted")))
        self.assertFalse(llm_call._looks_like_thinking_param_error(Exception("network timeout")))

    @covers(
        "utils.llm_call._invoke_with_optional_thinking_retry",
    )
    def test_retry_then_success_for_thinking_params(self):
        """Validate retry path when provider rejects thinking params.

        Args:
            self: Test case instance.
        """
        response = SimpleNamespace(
            choices = [SimpleNamespace(message = SimpleNamespace(content = "ok", tool_calls = None))],
            id = "resp_retry",
            model = "m_retry",
            usage = None,
            model_dump = lambda: {"id": "resp_retry"},
        )

        def _handler(request: Dict[str, Any]) -> Any:
            """Raise once for thinking param, then return success.

            Args:
                request: Completion request payload.
            """
            if request.get("enable_thinking") is True:
                raise Exception("unexpected keyword: enable_thinking")
            return response

        completions = _MockCompletions(handler = _handler)
        client = _MockClient(completions = completions)

        result = call_chat_completion(
            client = client,
            model = "m_retry",
            messages = [{"role": "user", "content": "hi"}],
            thinking_params = {"enable_thinking": True, "reasoning_effort": "high"},
        )

        self.assertEqual(len(completions.calls), 2)
        self.assertTrue(result.raw_metadata["thinking_params_stripped_retry"])
        self.assertIn("enable_thinking", result.raw_metadata["thinking_retry_error"])

    @covers(
        "utils.llm_call._coerce_text",
        "utils.llm_call._read_obj",
        "utils.llm_call._safe_model_dump",
    )
    def test_low_level_coerce_read_and_dump_helpers(self):
        """Validate helper conversion/access functions and fallback serialization.

        Args:
            self: Test case instance.
        """
        text_from_dict = llm_call._coerce_text(value = {"text": "abc"})
        text_from_list = llm_call._coerce_text(value = ["a", {"content": "b"}, _TextLike(text = "c")])
        text_from_obj = llm_call._coerce_text(value = _TextLike(text = None, reasoning = "why"))

        from_dict = llm_call._read_obj(obj = {"k": 1}, key = "k")
        from_attr = llm_call._read_obj(obj = SimpleNamespace(v = 2), key = "v")
        from_none = llm_call._read_obj(obj = None, key = "x")

        raw_dict = llm_call._safe_model_dump(obj = {"ok": True})
        raw_fallback = llm_call._safe_model_dump(obj = _BrokenModelDump())

        self.assertEqual(text_from_dict, "abc")
        self.assertEqual(text_from_list, "abc")
        self.assertEqual(text_from_obj, "why")
        self.assertEqual(from_dict, 1)
        self.assertEqual(from_attr, 2)
        self.assertIsNone(from_none)
        self.assertEqual(raw_dict, {"ok": True})
        self.assertEqual(raw_fallback, "broken-model-dump")

    @covers(
        "utils.llm_call._invoke_with_optional_thinking_retry",
    )
    def test_invoke_with_optional_retry_reraises_non_thinking_errors(self):
        """Validate non-thinking errors are re-raised without retry masking.

        Args:
            self: Test case instance.
        """
        completions = _MockCompletions(handler = lambda _: (_ for _ in ()).throw(Exception("network down")))
        client = _MockClient(completions = completions)

        with self.assertRaises(Exception):
            llm_call._invoke_with_optional_thinking_retry(
                client = client,
                request = {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "enable_thinking": True,
                },
                stream = False,
                on_content_chunk = None,
                on_reasoning_chunk = None,
                thinking_params = {"enable_thinking": True},
            )


class TestEnvRealQueries(unittest.TestCase):
    """Integration tests that execute real queries when env mode is enabled."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare env-based client only for env/both mode.

        Args:
            cls: Test class.
        """
        mode = TEST_CONTEXT.get("mode", "mock")
        if mode not in {"env", "both"}:
            raise unittest.SkipTest("Real env queries are disabled in mock mode.")

        try:
            cls.env_runtime = build_env_client_and_model()
        except Exception as exc:
            raise unittest.SkipTest(str(exc)) from exc

        cls.real_queries = get_real_queries()
        if not cls.real_queries:
            raise unittest.SkipTest("No real queries provided for env mode.")

    @covers("utils.llm_call.call_chat_completion")
    def test_env_non_stream_real_query(self):
        """Run one real non-stream query using .env config.

        Args:
            self: Test case instance.
        """
        prompt = self.real_queries[0]
        result = call_chat_completion(
            client = self.env_runtime["client"],
            model = self.env_runtime["model"],
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 80,
            stream = False,
        )

        content = (result.assistant_content or "").strip()
        self.assertTrue(content)

        TEST_CONTEXT.setdefault("real_query_logs", []).append(
            {
                "test": "env_non_stream",
                "query": prompt,
                "response_preview": content[:160],
                "stream": False,
            }
        )

    @covers("utils.llm_call.call_chat_completion")
    def test_env_stream_real_query(self):
        """Run one real stream query using .env config.

        Args:
            self: Test case instance.
        """
        prompt_index = 1 if len(self.real_queries) > 1 else 0
        prompt = self.real_queries[prompt_index]

        content_chunks: List[str] = []
        result = call_chat_completion(
            client = self.env_runtime["client"],
            model = self.env_runtime["model"],
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 80,
            stream = True,
            on_content_chunk = content_chunks.append,
            on_reasoning_chunk = None,
        )

        content = (result.assistant_content or "").strip()
        self.assertTrue(content)
        self.assertTrue(result.raw_metadata.get("stream"))
        self.assertGreaterEqual(result.raw_metadata.get("chunk_count", 0), 1)

        TEST_CONTEXT.setdefault("real_query_logs", []).append(
            {
                "test": "env_stream",
                "query": prompt,
                "response_preview": content[:160],
                "stream": True,
                "chunk_count": result.raw_metadata.get("chunk_count", 0),
                "content_chunks": len(content_chunks),
            }
        )


class TestUtilsExports(unittest.TestCase):
    """Tests for public utils export stability."""

    def test_export_set_stable(self):
        """Validate exported symbols remain expected.

        Args:
            self: Test case instance.
        """
        expected_exports = {
            "RuntimeOptions",
            "add_runtime_args",
            "runtime_options_from_args",
            "ThinkingPolicyState",
            "build_thinking_params",
            "resolve_thinking_policy",
            "LLMCallResult",
            "build_assistant_message",
            "call_chat_completion",
            "ReasoningRenderer",
            "TraceLogger",
            "SessionStore",
        }
        self.assertEqual(set(utils_exports), expected_exports)
        self.assertEqual(len(utils_exports), len(expected_exports))


class TestFunctionMatrix(unittest.TestCase):
    """Meta test that each discovered utils function has explicit coverage mapping."""

    def test_every_discovered_function_has_cover_annotation(self):
        """Validate no utils function is missing from @covers declarations.

        Args:
            self: Test case instance.
        """
        expected_targets = set(discover_function_targets(modules = TARGET_MODULES))
        suite = unittest.defaultTestLoader.loadTestsFromModule(module = sys.modules[__name__])

        declared_targets = set()
        for case in iter_test_cases(suite = suite):
            declared_targets.update(extract_covers(test_case = case))

        missing = sorted(expected_targets - declared_targets)
        self.assertEqual(missing, [])


def filter_suite_by_pattern(suite: unittest.TestSuite, pattern: str) -> unittest.TestSuite:
    """Filter test cases by substring pattern on unittest IDs.

    Args:
        suite: Original suite.
        pattern: Substring filter.
    """
    if not pattern:
        return suite

    filtered = unittest.TestSuite()
    for case in iter_test_cases(suite = suite):
        if pattern in case.id():
            filtered.addTest(case)

    return filtered


def resolve_target_status(statuses: List[str]) -> str:
    """Resolve one function target status from all linked test statuses.

    Args:
        statuses: Status list from related tests.
    """
    if not statuses:
        return "uncovered"
    if any(status in {"failure", "error"} for status in statuses):
        return "failing"
    if any(status == "success" for status in statuses):
        return "passed"
    if all(status == "skipped" for status in statuses):
        return "skipped"
    return "unknown"


def build_function_coverage(
    expected_targets: List[str],
    test_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build per-function completion map from recorded test outcomes.

    Args:
        expected_targets: Discovered function target list.
        test_records: Per-test result records.
    """
    raw_mapping: Dict[str, Dict[str, Any]] = {
        target: {"tests": [], "statuses": []}
        for target in expected_targets
    }
    declared_unknown_targets: Dict[str, Dict[str, Any]] = {}

    for record in test_records:
        for target in record.get("covers", []):
            if target not in raw_mapping:
                holder = declared_unknown_targets.setdefault(target, {"tests": [], "statuses": []})
                holder["tests"].append(record["name"])
                holder["statuses"].append(record["status"])
                continue

            raw_mapping[target]["tests"].append(record["name"])
            raw_mapping[target]["statuses"].append(record["status"])

    function_completion: Dict[str, Dict[str, Any]] = {}
    uncovered_targets: List[str] = []
    failing_targets: List[str] = []

    for target in expected_targets:
        linked_tests = raw_mapping[target]["tests"]
        statuses = raw_mapping[target]["statuses"]
        status = resolve_target_status(statuses = statuses)

        if status == "uncovered":
            uncovered_targets.append(target)
        if status == "failing":
            failing_targets.append(target)

        function_completion[target] = {
            "status": status,
            "tests": linked_tests,
        }

    return {
        "function_completion": function_completion,
        "uncovered_targets": uncovered_targets,
        "failing_targets": failing_targets,
        "declared_unknown_targets": declared_unknown_targets,
    }


def write_json_report(output_path: Path, report: Dict[str, Any]) -> None:
    """Write JSON report to disk.

    Args:
        output_path: Report file path.
        report: Report object.
    """
    output_path.parent.mkdir(parents = True, exist_ok = True)
    output_path.write_text(
        json.dumps(report, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )


def run_with_args(args: argparse.Namespace) -> int:
    """Run configured test flow and return process exit code.

    Args:
        args: Parsed CLI arguments.
    """
    TEST_CONTEXT["mode"] = args.mode
    TEST_CONTEXT["strict_coverage"] = bool(args.strict_coverage)
    TEST_CONTEXT["real_queries"] = list(args.real_queries or [])
    TEST_CONTEXT["real_query_logs"] = []

    loaded_dotenv: Dict[str, str] = {}
    if args.load_dotenv:
        loaded_dotenv = load_dotenv_file(dotenv_path = Path(args.dotenv_path))
    TEST_CONTEXT["dotenv_loaded"] = loaded_dotenv

    expected_targets = discover_function_targets(modules = TARGET_MODULES)

    if args.list_functions:
        for target in expected_targets:
            print(target)
        return 0

    root_suite = unittest.defaultTestLoader.loadTestsFromModule(module = sys.modules[__name__])
    selected_suite = filter_suite_by_pattern(suite = root_suite, pattern = args.pattern)

    if args.list_tests:
        for case in iter_test_cases(suite = selected_suite):
            print(case.id())
        return 0

    runner = unittest.TextTestRunner(
        verbosity = args.verbosity,
        resultclass = RecordingTextResult,
    )
    result: RecordingTextResult = runner.run(selected_suite)

    test_records = [
        result.records[key]
        for key in sorted(result.records.keys())
    ]

    coverage = build_function_coverage(
        expected_targets = expected_targets,
        test_records = test_records,
    )

    all_tests_passed = result.wasSuccessful()
    strict_coverage = bool(args.strict_coverage)
    has_uncovered = bool(coverage["uncovered_targets"])
    has_failing_targets = bool(coverage["failing_targets"])
    has_unknown_declared = bool(coverage["declared_unknown_targets"])

    complete = all_tests_passed and not has_failing_targets
    if strict_coverage:
        complete = complete and (not has_uncovered) and (not has_unknown_declared)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "dotenv_path": str(Path(args.dotenv_path)),
        "dotenv_loaded_keys": sorted(list(loaded_dotenv.keys())),
        "real_queries_requested": get_real_queries() if args.mode in {"env", "both"} else [],
        "real_query_logs": list(TEST_CONTEXT.get("real_query_logs", [])),
        "pattern": args.pattern,
        "strict_coverage": strict_coverage,
        "summary": {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "all_tests_passed": all_tests_passed,
            "uncovered_function_count": len(coverage["uncovered_targets"]),
            "failing_function_count": len(coverage["failing_targets"]),
            "declared_unknown_target_count": len(coverage["declared_unknown_targets"]),
            "complete": complete,
        },
        "tests": test_records,
        "function_completion": coverage["function_completion"],
        "uncovered_targets": coverage["uncovered_targets"],
        "failing_targets": coverage["failing_targets"],
        "declared_unknown_targets": coverage["declared_unknown_targets"],
    }

    output_path = Path(args.output_json)
    write_json_report(output_path = output_path, report = report)

    print(f"JSON report written: {output_path}")
    return 0 if complete else 1


def main() -> int:
    """CLI entrypoint for local test execution.

    Args:
        None.
    """
    args = parse_args()
    return run_with_args(args = args)


if __name__ == "__main__":
    raise SystemExit(main())
