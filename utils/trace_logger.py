"""Per-turn LLM response trace logger."""

import json
import logging
from typing import Any, Dict, List, Optional


class TraceLogger:
    """Conditional trace logging for assistant replies and tool calls."""

    def __init__(self, enabled: bool, logger: Optional[logging.Logger] = None):
        self.enabled = bool(enabled)
        self.logger = logger or logging.getLogger("TraceLogger")

    def log_turn(
        self,
        actor: str,
        assistant_content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        assistant_reasoning: str = "",
    ) -> None:
        """Log assistant text, tool-call summaries, and reasoning preview."""
        if not self.enabled:
            return

        content_preview = _shorten(assistant_content or "", 400)
        self.logger.info(f"[LLM:{actor}] assistant: {content_preview or '(empty)'}")

        if tool_calls:
            summary = "; ".join(_summarize_tool_call(tool_call) for tool_call in tool_calls)
            self.logger.info(f"[LLM:{actor}] tool_calls: {summary}")

        reasoning_preview = _shorten(assistant_reasoning or "", 200)
        if reasoning_preview:
            self.logger.info(f"[LLM:{actor}] reasoning: {reasoning_preview}")


def _summarize_tool_call(tool_call: Dict[str, Any]) -> str:
    """Build compact 'name(args)' summary from a tool call payload."""
    function_block = tool_call.get("function") or {}
    tool_name = function_block.get("name") or "unknown"
    arguments = function_block.get("arguments") or "{}"

    args_preview = arguments
    try:
        parsed = json.loads(arguments)
        args_preview = json.dumps(parsed, ensure_ascii = False)
    except Exception:
        args_preview = str(arguments)

    return f"{tool_name}({_shorten(args_preview, 160)})"


def _shorten(text: str, max_chars: int) -> str:
    """Trim long text for concise logs."""
    normalized = text.replace("\n", "\\n").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."
