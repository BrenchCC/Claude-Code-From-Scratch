"""Session persistence for assistant/tool events in JSONL format."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class SessionStore:
    """Append-only JSONL session logger with model+timestamp naming.

    Args:
        enabled: Whether persistence is enabled.
        model: Model identifier used in file naming.
        session_dir: Session output directory.
        runtime_options: Optional runtime metadata snapshot.
    """

    def __init__(
        self,
        enabled: bool,
        model: str,
        session_dir: Path,
        runtime_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize session store.

        Args:
            enabled: Whether persistence is enabled.
            model: Model identifier used in file naming.
            session_dir: Session output directory.
            runtime_options: Optional runtime metadata snapshot.
        """
        self.enabled = bool(enabled)
        self.model = model or "unknown-model"
        self.session_dir = Path(session_dir)
        self.runtime_options = runtime_options or {}
        self.path: Optional[Path] = None

        if self.enabled:
            self._initialize_output_file()
            self._append(
                {
                    "event": "meta",
                    "timestamp": _now_iso(),
                    "model": self.model,
                    "runtime_options": self.runtime_options,
                }
            )

    def record_assistant(
        self,
        actor: str,
        content: str,
        reasoning: str,
        tool_calls: Any,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one assistant event.

        Args:
            actor: Actor identifier.
            content: Assistant content text.
            reasoning: Assistant reasoning text.
            tool_calls: Tool calls payload.
            raw_metadata: Optional raw metadata.
        """
        self._append(
            self._build_event(
                event = "assistant",
                actor = actor,
                content = content or "",
                reasoning = reasoning or "",
                tool_calls = tool_calls or [],
                raw_metadata = raw_metadata or {},
            )
        )

    def record_tool(
        self,
        actor: str,
        tool_name: str,
        arguments: Dict[str, Any],
        output: Any,
    ) -> None:
        """Record one tool execution event.

        Args:
            actor: Actor identifier.
            tool_name: Tool name.
            arguments: Tool arguments.
            output: Tool output.
        """
        self._append(
            self._build_event(
                event = "tool",
                actor = actor,
                tool_name = tool_name,
                arguments = arguments,
                output = output,
            )
        )

    def get_path(self) -> Optional[Path]:
        """Return output file path when session saving is enabled.

        Args:
            None.
        """
        return self.path

    def _initialize_output_file(self) -> None:
        """Create output directory and compute session file path.

        Args:
            None.
        """
        self.session_dir.mkdir(parents = True, exist_ok = True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_model = _sanitize_model_name(self.model)
        self.path = self.session_dir / f"{sanitized_model}_{timestamp}.jsonl"

    def _build_event(self, event: str, **payload: Any) -> Dict[str, Any]:
        """Build one timestamped event payload.

        Args:
            event: Event kind.
            **payload: Event-specific payload fields.
        """
        return {
            "event": event,
            "timestamp": _now_iso(),
            **payload,
        }

    def _append(self, payload: Dict[str, Any]) -> None:
        """Append one JSON line if persistence is enabled.

        Args:
            payload: Event payload.
        """
        # Early-return keeps disabled mode side-effect free.
        if not self.enabled or self.path is None:
            return

        with self.path.open("a", encoding = "utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii = False) + "\n")


def _sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for filesystem-safe session filename.

    Args:
        model_name: Raw model name.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip())
    return sanitized.strip("_") or "unknown-model"


def _now_iso() -> str:
    """Return current local timestamp in ISO-like format.

    Args:
        None.
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
