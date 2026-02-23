"""Runtime option parsing shared by v1-v5 agents."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BOOL_TRUE = {"1", "true", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "no", "n", "off"}


@dataclass
class RuntimeOptions:
    """Runtime feature switches merged from CLI and environment variables."""

    show_llm_response: bool = False
    stream: bool = False
    thinking_mode: str = "auto"
    reasoning_effort: str = "none"
    reasoning_preview_chars: int = 200
    save_session: bool = False
    session_dir: Path = Path("sessions")
    thinking_capability: str = "auto"
    thinking_param_style: str = "auto"

    def as_dict(self) -> dict:
        """Return JSON-serializable dict form for session metadata."""
        return {
            "show_llm_response": self.show_llm_response,
            "stream": self.stream,
            "thinking_mode": self.thinking_mode,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_preview_chars": self.reasoning_preview_chars,
            "save_session": self.save_session,
            "session_dir": str(self.session_dir),
            "thinking_capability": self.thinking_capability,
            "thinking_param_style": self.thinking_param_style,
        }


def add_runtime_args(parser: Any) -> None:
    """Attach shared runtime flags to an argparse parser."""
    import argparse

    parser.add_argument(
        "--show-llm-response",
        dest = "show_llm_response",
        action = argparse.BooleanOptionalAction,
        default = None,
        help = "Show per-turn LLM assistant/tool/reasoning trace logs.",
    )
    parser.add_argument(
        "--stream",
        dest = "stream",
        action = argparse.BooleanOptionalAction,
        default = None,
        help = "Enable streaming output from the model.",
    )
    parser.add_argument(
        "--thinking",
        dest = "thinking",
        choices = ["auto", "on", "off"],
        default = None,
        help = "Thinking display/control mode.",
    )
    parser.add_argument(
        "--reasoning-effort",
        dest = "reasoning_effort",
        choices = ["none", "low", "medium", "high"],
        default = None,
        help = "Requested reasoning effort level.",
    )
    parser.add_argument(
        "--reasoning-preview-chars",
        dest = "reasoning_preview_chars",
        type = int,
        default = None,
        help = "Preview char count before folding reasoning in stream mode.",
    )
    parser.add_argument(
        "--save-session",
        dest = "save_session",
        action = argparse.BooleanOptionalAction,
        default = None,
        help = "Save conversation events to JSONL.",
    )
    parser.add_argument(
        "--session-dir",
        dest = "session_dir",
        default = None,
        help = "Session output directory (default: sessions/).",
    )


def runtime_options_from_args(args: Any) -> RuntimeOptions:
    """Build runtime options with CLI > ENV > default precedence."""
    show_llm_response = _resolve_bool(
        cli_value = getattr(args, "show_llm_response", None),
        env_name = "AGENT_SHOW_LLM_RESPONSE",
        default = False,
    )
    stream = _resolve_bool(
        cli_value = getattr(args, "stream", None),
        env_name = "AGENT_STREAM",
        default = False,
    )
    thinking_mode = _resolve_enum(
        cli_value = getattr(args, "thinking", None),
        env_name = "AGENT_THINKING_MODE",
        default = "auto",
        allowed = {"auto", "on", "off"},
    )
    reasoning_effort = _resolve_enum(
        cli_value = getattr(args, "reasoning_effort", None),
        env_name = "AGENT_REASONING_EFFORT",
        default = "none",
        allowed = {"none", "low", "medium", "high"},
    )
    reasoning_preview_chars = _resolve_int(
        cli_value = getattr(args, "reasoning_preview_chars", None),
        env_name = "AGENT_REASONING_PREVIEW_CHARS",
        default = 200,
    )
    save_session = _resolve_bool(
        cli_value = getattr(args, "save_session", None),
        env_name = "AGENT_SAVE_SESSION",
        default = False,
    )

    raw_session_dir = _resolve_str(
        cli_value = getattr(args, "session_dir", None),
        env_name = "AGENT_SESSION_DIR",
        default = "sessions",
    )
    thinking_capability = _resolve_enum(
        cli_value = None,
        env_name = "AGENT_THINKING_CAPABILITY",
        default = "auto",
        allowed = {"auto", "toggle", "always", "never"},
    )
    thinking_param_style = _resolve_enum(
        cli_value = None,
        env_name = "AGENT_THINKING_PARAM_STYLE",
        default = "auto",
        allowed = {"auto", "enable_thinking", "reasoning_effort", "both"},
    )

    return RuntimeOptions(
        show_llm_response = show_llm_response,
        stream = stream,
        thinking_mode = thinking_mode,
        reasoning_effort = reasoning_effort,
        reasoning_preview_chars = max(0, reasoning_preview_chars),
        save_session = save_session,
        session_dir = Path(raw_session_dir),
        thinking_capability = thinking_capability,
        thinking_param_style = thinking_param_style,
    )


def _resolve_bool(cli_value: Any, env_name: str, default: bool) -> bool:
    """Resolve bool with CLI > ENV > default precedence."""
    if cli_value is not None:
        return bool(cli_value)

    raw_env = os.getenv(env_name)
    if raw_env is None:
        return default

    normalized = raw_env.strip().lower()
    if normalized in BOOL_TRUE:
        return True
    if normalized in BOOL_FALSE:
        return False
    return default


def _resolve_enum(cli_value: Any, env_name: str, default: str, allowed: set) -> str:
    """Resolve enum option with validation."""
    if cli_value is not None and str(cli_value) in allowed:
        return str(cli_value)

    raw_env = os.getenv(env_name)
    if raw_env is not None:
        normalized = raw_env.strip().lower()
        if normalized in allowed:
            return normalized

    return default


def _resolve_int(cli_value: Any, env_name: str, default: int) -> int:
    """Resolve int option with fallback to default on parse failure."""
    if cli_value is not None:
        return int(cli_value)

    raw_env = os.getenv(env_name)
    if raw_env is None:
        return default

    try:
        return int(raw_env.strip())
    except ValueError:
        return default


def _resolve_str(cli_value: Any, env_name: str, default: str) -> str:
    """Resolve string option with CLI > ENV > default precedence."""
    if cli_value is not None and str(cli_value).strip():
        return str(cli_value)

    raw_env = os.getenv(env_name)
    if raw_env is not None and raw_env.strip():
        return raw_env.strip()

    return default
