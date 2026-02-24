"""Regression tests for utils refactor with behavior compatibility checks."""

import io
import json
import logging
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock, patch


sys.path.append(os.getcwd())

from utils import __all__ as utils_exports
from utils.llm_call import build_assistant_message, call_chat_completion
from utils.reasoning_renderer import ReasoningRenderer
from utils.runtime_config import RuntimeOptions, runtime_options_from_args
from utils.session_store import SessionStore
from utils.thinking_policy import ThinkingPolicyState, build_thinking_params, resolve_thinking_policy
from utils.trace_logger import TraceLogger


class _MockCompletions:
    """Mock completion endpoint with call capture.

    Args:
        handler: Callable that returns response or raises error.
    """

    def __init__(self, handler):
        """Initialize mock endpoint.

        Args:
            handler: Callable that receives request kwargs.
        """
        self._handler = handler
        self.calls = []

    def create(self, **kwargs):
        """Record request and delegate to handler.

        Args:
            **kwargs: Completion request kwargs.
        """
        self.calls.append(kwargs)
        return self._handler(kwargs)


class _MockClient:
    """Mock client exposing chat.completions.create.

    Args:
        completions: Mock completions endpoint.
    """

    def __init__(self, completions):
        """Initialize mock client.

        Args:
            completions: Mock completions endpoint.
        """
        self.chat = SimpleNamespace(completions = completions)


class _ProbeCompletions:
    """Probe endpoint stub for thinking-policy capability resolution.

    Args:
        supports_on: Whether "enabled" probes should succeed.
        supports_off: Whether "disabled" probes should succeed.
    """

    def __init__(self, supports_on: bool, supports_off: bool):
        """Initialize probe endpoint.

        Args:
            supports_on: Whether "enabled" probes should succeed.
            supports_off: Whether "disabled" probes should succeed.
        """
        self.supports_on = supports_on
        self.supports_off = supports_off

    def create(self, **kwargs):
        """Simulate provider probe behavior.

        Args:
            **kwargs: Completion request kwargs.
        """
        params = {
            key: value
            for key, value in kwargs.items()
            if key in {"enable_thinking", "reasoning_effort"}
        }
        enabled = False
        if params.get("enable_thinking") is True:
            enabled = True
        if params.get("reasoning_effort") not in {None, "none"}:
            enabled = True

        if enabled and not self.supports_on:
            raise Exception("unsupported")
        if (not enabled) and not self.supports_off:
            raise Exception("unsupported")

        return SimpleNamespace(ok = True)


class TestRuntimeConfig(unittest.TestCase):
    """Tests for runtime option resolution precedence."""

    def test_cli_overrides_environment(self):
        """Validate CLI values win over environment values.

        Args:
            self: Test case instance.
        """
        args = Namespace(
            show_llm_response = True,
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
                "AGENT_SHOW_LLM_RESPONSE": "false",
                "AGENT_STREAM": "true",
            },
            clear = True,
        ):
            options = runtime_options_from_args(args = args)

        self.assertTrue(options.show_llm_response)
        self.assertTrue(options.stream)

    def test_invalid_environment_values_fallback(self):
        """Validate invalid env values fall back to defaults.

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
                "AGENT_STREAM": "not_bool",
                "AGENT_REASONING_PREVIEW_CHARS": "invalid_int",
                "AGENT_REASONING_EFFORT": "HIGH",
                "AGENT_SESSION_DIR": "my_sessions",
            },
            clear = True,
        ):
            options = runtime_options_from_args(args = args)

        self.assertFalse(options.stream)
        self.assertEqual(options.reasoning_preview_chars, 200)
        self.assertEqual(options.reasoning_effort, "high")
        self.assertEqual(str(options.session_dir), "my_sessions")

    def test_runtime_options_as_dict(self):
        """Validate RuntimeOptions.as_dict serialization keys.

        Args:
            self: Test case instance.
        """
        options = RuntimeOptions(
            show_llm_response = True,
            stream = True,
            thinking_mode = "on",
            reasoning_effort = "medium",
        )
        payload = options.as_dict()

        self.assertTrue(payload["show_llm_response"])
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["thinking_mode"], "on")
        self.assertEqual(payload["reasoning_effort"], "medium")


class TestThinkingPolicy(unittest.TestCase):
    """Tests for thinking capability resolution and params."""

    def test_manual_capability_uses_default_style(self):
        """Validate manual capability keeps compatibility style default.

        Args:
            self: Test case instance.
        """
        state = resolve_thinking_policy(
            client = None,
            model = "",
            capability_setting = "always",
            param_style_setting = "auto",
        )

        self.assertEqual(state.capability, "always")
        self.assertEqual(state.param_style, "enable_thinking")

    def test_probe_toggle_resolution(self):
        """Validate probe resolves toggle capability when both probes pass.

        Args:
            self: Test case instance.
        """
        client = _MockClient(
            completions = _ProbeCompletions(
                supports_on = True,
                supports_off = True,
            )
        )
        state = resolve_thinking_policy(client = client, model = "model")

        self.assertEqual(state.capability, "toggle")
        self.assertEqual(state.param_style, "enable_thinking")

    def test_probe_always_resolution(self):
        """Validate probe resolves always capability when only enabled probe passes.

        Args:
            self: Test case instance.
        """
        client = _MockClient(
            completions = _ProbeCompletions(
                supports_on = True,
                supports_off = False,
            )
        )
        state = resolve_thinking_policy(client = client, model = "model")

        self.assertEqual(state.capability, "always")
        self.assertEqual(state.param_style, "enable_thinking")

    def test_build_thinking_params_for_toggle(self):
        """Validate mode on/off mapping for toggle policies.

        Args:
            self: Test case instance.
        """
        policy = ThinkingPolicyState(capability = "toggle", param_style = "both")

        params_on = build_thinking_params(
            policy = policy,
            thinking_mode = "on",
            reasoning_effort = "high",
        )
        params_off = build_thinking_params(
            policy = policy,
            thinking_mode = "off",
            reasoning_effort = "high",
        )

        self.assertEqual(params_on, {"enable_thinking": True, "reasoning_effort": "high"})
        self.assertEqual(params_off, {"enable_thinking": False, "reasoning_effort": "none"})

    def test_build_thinking_params_returns_empty_when_not_applicable(self):
        """Validate unsupported mode/capability combinations return empty params.

        Args:
            self: Test case instance.
        """
        policy = ThinkingPolicyState(capability = "always", param_style = "enable_thinking")
        params = build_thinking_params(
            policy = policy,
            thinking_mode = "off",
            reasoning_effort = "high",
        )

        self.assertEqual(params, {})


class TestLLMCall(unittest.TestCase):
    """Tests for LLM call normalization and retry behavior."""

    def test_non_stream_normalization(self):
        """Validate non-stream response normalization.

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

        self.assertEqual(result.assistant_content, "hello")
        self.assertEqual(result.assistant_reasoning, "r1r2r3")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "tool_a")
        self.assertEqual(result.raw_metadata["stream"], False)

    def test_stream_normalization_and_callbacks(self):
        """Validate stream merge for content, reasoning, and tool calls.

        Args:
            self: Test case instance.
        """
        chunk_1 = SimpleNamespace(
            id = "chunk_1",
            model = "model_1",
            choices = [
                SimpleNamespace(
                    delta = SimpleNamespace(
                        content = "Hel",
                        reasoning = "R1",
                        tool_calls = [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "fo", "arguments": '{"x"'},
                            }
                        ],
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
                            {"type": "reasoning", "text": "ignored_for_content"},
                        ],
                        content_extra = "",
                        thinking = "R2",
                        tool_calls = [
                            {
                                "index": 0,
                                "function": {"name": "o", "arguments": ':1}'},
                            }
                        ],
                    )
                )
            ],
        )

        completions = _MockCompletions(handler = lambda _: [chunk_1, chunk_2])
        client = _MockClient(completions = completions)

        content_chunks = []
        reasoning_chunks = []
        result = call_chat_completion(
            client = client,
            model = "model_1",
            messages = [{"role": "user", "content": "hi"}],
            stream = True,
            on_content_chunk = content_chunks.append,
            on_reasoning_chunk = reasoning_chunks.append,
        )

        self.assertEqual(result.assistant_content, "Hello")
        self.assertEqual(result.assistant_reasoning, "R1R2ignored_for_content")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "fo")
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], '{"x":1}')
        self.assertEqual(content_chunks, ["Hel", "lo"])
        self.assertEqual(reasoning_chunks, ["R1", "R2ignored_for_content"])
        self.assertEqual(result.raw_metadata["stream"], True)
        self.assertEqual(result.raw_metadata["chunk_count"], 2)

    def test_thinking_param_retry(self):
        """Validate retry strips unsupported thinking params once.

        Args:
            self: Test case instance.
        """
        message = SimpleNamespace(content = "ok", tool_calls = None)
        response = SimpleNamespace(
            choices = [SimpleNamespace(message = message)],
            id = "resp_2",
            model = "model_2",
            usage = None,
            model_dump = lambda: {"id": "resp_2"},
        )

        def _handler(request):
            """Raise on first call and return success on second call.

            Args:
                request: Completion request kwargs.
            """
            if request.get("enable_thinking") is True:
                raise Exception("unknown parameter: enable_thinking")
            return response

        completions = _MockCompletions(handler = _handler)
        client = _MockClient(completions = completions)

        result = call_chat_completion(
            client = client,
            model = "model_2",
            messages = [{"role": "user", "content": "hi"}],
            thinking_params = {"enable_thinking": True, "reasoning_effort": "high"},
        )

        self.assertEqual(len(completions.calls), 2)
        self.assertTrue(result.raw_metadata["thinking_params_stripped_retry"])
        self.assertIn("unknown parameter", result.raw_metadata["thinking_retry_error"])

    def test_build_assistant_message(self):
        """Validate assistant message conversion with tool calls.

        Args:
            self: Test case instance.
        """
        result = SimpleNamespace(
            assistant_content = "hello",
            tool_calls = [{"id": "1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        )
        message = build_assistant_message(result = result)

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "hello")
        self.assertEqual(len(message["tool_calls"]), 1)


class TestReasoningRenderer(unittest.TestCase):
    """Tests for reasoning preview/fold/expand renderer."""

    def test_stream_preview_and_fold(self):
        """Validate preview accumulation and fold notification.

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
        self.assertEqual(result_b["preview_appended"], "de")
        self.assertTrue(result_b["folded"])
        self.assertIn("[Reasoning Preview]", output.getvalue())
        self.assertIn("[Reasoning Folded]", output.getvalue())

    def test_finalize_with_expand(self):
        """Validate non-stream hint and manual expand path.

        Args:
            self: Test case instance.
        """
        output = io.StringIO()
        renderer = ReasoningRenderer(
            preview_chars = 10,
            output_stream = output,
            input_func = lambda _: "r",
        )

        result = renderer.finalize_turn(
            full_reasoning = "full reasoning text",
            stream_mode = False,
            allow_expand_prompt = True,
            interactive = True,
        )

        self.assertTrue(result["expanded"])
        self.assertIn("[Reasoning Available]", output.getvalue())
        self.assertIn("[Reasoning Expanded]", output.getvalue())


class TestSessionStore(unittest.TestCase):
    """Tests for JSONL session persistence behavior."""

    def test_enabled_store_writes_meta_and_events(self):
        """Validate enabled store writes meta, assistant, and tool events.

        Args:
            self: Test case instance.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = os.path.join(temp_dir, "sessions")
            store = SessionStore(
                enabled = True,
                model = "test/model",
                session_dir = session_dir,
                runtime_options = {"stream": True},
            )

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

            with path.open("r", encoding = "utf-8") as file:
                rows = [json.loads(line) for line in file]

        self.assertEqual(rows[0]["event"], "meta")
        self.assertEqual(rows[1]["event"], "assistant")
        self.assertEqual(rows[2]["event"], "tool")

    def test_disabled_store_has_no_side_effects(self):
        """Validate disabled store does not create files.

        Args:
            self: Test case instance.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = os.path.join(temp_dir, "sessions")
            store = SessionStore(
                enabled = False,
                model = "test",
                session_dir = session_dir,
            )

            store.record_assistant(
                actor = "assistant",
                content = "hello",
                reasoning = "",
                tool_calls = [],
            )

            self.assertIsNone(store.get_path())
            self.assertFalse(os.path.exists(session_dir))


class TestTraceLogger(unittest.TestCase):
    """Tests for trace logging previews and disabled mode."""

    def test_log_turn_enabled(self):
        """Validate log message content in enabled mode.

        Args:
            self: Test case instance.
        """
        logger = logging.getLogger("trace_logger_test")
        logger.handlers = []

        trace_logger = TraceLogger(enabled = True, logger = logger)

        with self.assertLogs(logger, level = "INFO") as captured:
            trace_logger.log_turn(
                actor = "agent",
                assistant_content = "x" * 500,
                tool_calls = [
                    {
                        "function": {
                            "name": "tool_x",
                            "arguments": '{"a":1}',
                        }
                    }
                ],
                assistant_reasoning = "r" * 300,
            )

        combined = "\n".join(captured.output)
        self.assertIn("[LLM:agent] assistant:", combined)
        self.assertIn("[LLM:agent] tool_calls:", combined)
        self.assertIn("[LLM:agent] reasoning:", combined)

    def test_log_turn_disabled(self):
        """Validate logger is not called when disabled.

        Args:
            self: Test case instance.
        """
        mock_logger = Mock()
        trace_logger = TraceLogger(enabled = False, logger = mock_logger)

        trace_logger.log_turn(
            actor = "agent",
            assistant_content = "text",
            tool_calls = [],
            assistant_reasoning = "reasoning",
        )

        mock_logger.info.assert_not_called()


class TestUtilsExports(unittest.TestCase):
    """Tests for public utils export stability."""

    def test_export_set_stable(self):
        """Validate public export symbols remain unchanged.

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


if __name__ == "__main__":
    unittest.main()
