"""Per-turn LLM response trace logger."""

import json
import logging
from typing import Any, Dict, List, Optional


_CONTENT_PREVIEW_LIMIT = 400
_REASONING_PREVIEW_LIMIT = 200
_TOOL_ARGS_PREVIEW_LIMIT = 160


class TraceLogger:
    """Conditional trace logging for assistant replies and tool calls.

    Args:
        enabled: Whether trace logging is enabled.
        logger: Optional logger instance.
    """

    def __init__(self, enabled: bool, logger: Optional[logging.Logger] = None):
        """Initialize trace logger.

        Args:
            enabled: Whether trace logging is enabled.
            logger: Optional logger instance.
        """
        self.enabled = bool(enabled)
        self.logger = logger or logging.getLogger("TraceLogger")

    def log_turn(
        self,
        actor: str,
        assistant_content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        assistant_reasoning: str = "",
    ) -> None:
        """Log assistant text, tool-call summaries, and reasoning preview.

        Args:
            actor: Actor identifier.
            assistant_content: Assistant plain content.
            tool_calls: Optional normalized tool call list.
            assistant_reasoning: Assistant reasoning text.
        """
        if not self.enabled:
            return

        content_preview = _shorten(assistant_content or "", _CONTENT_PREVIEW_LIMIT)
        self.logger.info(f"[LLM:{actor}] assistant: {content_preview or '(empty)'}")

        if tool_calls:
            summary = "; ".join(_summarize_tool_call(tool_call) for tool_call in tool_calls)
            self.logger.info(f"[LLM:{actor}] tool_calls: {summary}")

        reasoning_preview = _shorten(assistant_reasoning or "", _REASONING_PREVIEW_LIMIT)
        if reasoning_preview:
            self.logger.info(f"[LLM:{actor}] reasoning: {reasoning_preview}")


def _summarize_tool_call(tool_call: Dict[str, Any]) -> str:
    """Build compact 'name(args)' summary from a tool call payload.

    Args:
        tool_call: One normalized tool call.
    """
    function_block = tool_call.get("function") or {}
    tool_name = function_block.get("name") or "unknown"
    arguments = function_block.get("arguments") or "{}"

    args_preview = arguments
    try:
        parsed = json.loads(arguments)
        args_preview = json.dumps(parsed, ensure_ascii = False)
    except Exception:
        args_preview = str(arguments)

    return f"{tool_name}({_shorten(args_preview, _TOOL_ARGS_PREVIEW_LIMIT)})"


def _shorten(text: str, max_chars: int) -> str:
    """Trim long text for concise logs.

    Args:
        text: Input text.
        max_chars: Maximum preview length.
    """
    normalized = text.replace("\n", "\\n").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."
