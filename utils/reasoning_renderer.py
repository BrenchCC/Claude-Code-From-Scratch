"""Reasoning preview/fold/expand renderer for terminal flows."""

import sys
from typing import Any, Callable, Dict


class ReasoningRenderer:
    """Render reasoning preview with fold + optional on-demand expansion.

    Args:
        preview_chars: Preview character budget in stream mode.
        output_stream: Stream target for rendering output.
        input_func: Input function used for expand prompt.
    """

    def __init__(
        self,
        preview_chars: int = 200,
        output_stream = None,
        input_func: Callable[[str], str] = input,
    ):
        """Initialize renderer state.

        Args:
            preview_chars: Preview character budget in stream mode.
            output_stream: Stream target for rendering output.
            input_func: Input function used for expand prompt.
        """
        self.preview_chars = max(0, int(preview_chars))
        self.output_stream = output_stream or sys.stdout
        self.input_func = input_func
        self.reset_turn()

    def reset_turn(self) -> None:
        """Reset stream rendering state for a new assistant turn.

        Args:
            None.
        """
        self._reasoning_buffer = ""
        self._preview_printed = 0
        self._preview_started = False
        self._folded = False
        self._fold_notice_printed = False

    def handle_stream_chunk(self, chunk: str) -> Dict[str, Any]:
        """Render a streaming reasoning chunk using preview + fold policy.

        Args:
            chunk: Newly received reasoning fragment.
        """
        if not chunk:
            return {
                "preview_appended": "",
                "folded": self._folded,
            }

        self._reasoning_buffer += chunk

        preview_appended = self._emit_preview(chunk = chunk)
        self._emit_fold_notice_if_needed()

        return {
            "preview_appended": preview_appended,
            "folded": self._folded,
        }

    def finalize_turn(
        self,
        full_reasoning: str,
        stream_mode: bool,
        allow_expand_prompt: bool,
        interactive: bool = True,
    ) -> Dict[str, Any]:
        """Finalize turn: show hint and optionally expand full reasoning by user input.

        Args:
            full_reasoning: Full final reasoning text.
            stream_mode: Whether this turn used streaming output.
            allow_expand_prompt: Whether to ask for expand action.
            interactive: Whether user interaction is allowed.
        """
        reasoning = full_reasoning or ""
        if not reasoning:
            return {
                "has_reasoning": False,
                "folded": False,
                "expanded": False,
                "hint": "",
                "full_reasoning": "",
            }

        hint = self._build_hint(reasoning = reasoning, stream_mode = stream_mode)
        if hint:
            if not stream_mode:
                self._write(f"\n{hint}\n")
            else:
                self._write(f"{hint}\n")

        expanded = False
        if allow_expand_prompt and interactive:
            expanded = self._maybe_expand(reasoning = reasoning)

        return {
            "has_reasoning": True,
            "folded": len(reasoning) > self.preview_chars,
            "expanded": expanded,
            "hint": hint,
            "full_reasoning": reasoning,
        }

    def _emit_preview(self, chunk: str) -> str:
        """Emit preview text from one reasoning chunk.

        Args:
            chunk: Newly received reasoning fragment.
        """
        remaining = self.preview_chars - self._preview_printed
        if remaining <= 0:
            return ""

        preview_appended = chunk[:remaining]
        if not preview_appended:
            return ""

        if not self._preview_started:
            self._write("\n[Reasoning Preview]\n")
            self._preview_started = True

        self._write(preview_appended)
        self._preview_printed += len(preview_appended)
        return preview_appended

    def _emit_fold_notice_if_needed(self) -> None:
        """Emit folded notice once when reasoning exceeds preview budget.

        Args:
            None.
        """
        if len(self._reasoning_buffer) <= self.preview_chars:
            return
        if self._fold_notice_printed:
            return

        if self._preview_started:
            self._write("\n")
        self._write("[Reasoning Folded] Preview limit reached; hidden reasoning is buffered.\n")
        self._folded = True
        self._fold_notice_printed = True

    def _build_hint(self, reasoning: str, stream_mode: bool) -> str:
        """Build per-turn reasoning availability hint text.

        Args:
            reasoning: Full final reasoning text.
            stream_mode: Whether this turn used streaming output.
        """
        if not stream_mode:
            return "[Reasoning Available] Input 'r' to expand full reasoning."
        if len(reasoning) > self.preview_chars:
            return "[Reasoning Available] Input 'r' to expand hidden reasoning."
        return ""

    def _maybe_expand(self, reasoning: str) -> bool:
        """Prompt user once for on-demand full reasoning expansion.

        Args:
            reasoning: Full final reasoning text.
        """
        try:
            user_input = self.input_func("Expand reasoning? [r/N]: ").strip().lower()
        except EOFError:
            return False
        except KeyboardInterrupt:
            return False

        if user_input != "r":
            return False

        self._write("[Reasoning Expanded]\n")
        self._write(reasoning)
        self._write("\n")
        return True

    def _write(self, text: str) -> None:
        """Write text to configured stream with immediate flush.

        Args:
            text: Output text.
        """
        self.output_stream.write(text)
        self.output_stream.flush()
