"""Reasoning preview/fold/expand renderer for terminal flows."""

import sys
from typing import Any, Callable, Dict


class ReasoningRenderer:
    """Render reasoning preview with fold + optional on-demand expansion."""

    def __init__(
        self,
        preview_chars: int = 200,
        output_stream = None,
        input_func: Callable[[str], str] = input,
    ):
        self.preview_chars = max(0, int(preview_chars))
        self.output_stream = output_stream or sys.stdout
        self.input_func = input_func
        self.reset_turn()

    def reset_turn(self) -> None:
        """Reset stream rendering state for a new assistant turn."""
        self._reasoning_buffer = ""
        self._preview_printed = 0
        self._preview_started = False
        self._folded = False
        self._fold_notice_printed = False

    def handle_stream_chunk(self, chunk: str) -> Dict[str, Any]:
        """Render a streaming reasoning chunk using preview + fold policy."""
        if not chunk:
            return {
                "preview_appended": "",
                "folded": self._folded,
            }

        self._reasoning_buffer += chunk

        preview_appended = ""
        remaining = self.preview_chars - self._preview_printed
        if remaining > 0:
            preview_appended = chunk[:remaining]
            if preview_appended:
                if not self._preview_started:
                    self._write("\n[Reasoning Preview]\n")
                    self._preview_started = True
                self._write(preview_appended)
                self._preview_printed += len(preview_appended)

        if len(self._reasoning_buffer) > self.preview_chars and not self._fold_notice_printed:
            if self._preview_started:
                self._write("\n")
            self._write("[Reasoning Folded] Preview limit reached; hidden reasoning is buffered.\n")
            self._folded = True
            self._fold_notice_printed = True

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
        """Finalize turn: show hint and optionally expand full reasoning by user input."""
        reasoning = full_reasoning or ""
        if not reasoning:
            return {
                "has_reasoning": False,
                "folded": False,
                "expanded": False,
                "hint": "",
                "full_reasoning": "",
            }

        if not stream_mode:
            hint = "[Reasoning Available] Input 'r' to expand full reasoning."
            self._write(f"\n{hint}\n")
        elif len(reasoning) > self.preview_chars:
            hint = "[Reasoning Available] Input 'r' to expand hidden reasoning."
            self._write(f"{hint}\n")
        else:
            hint = ""

        expanded = False
        if allow_expand_prompt and interactive:
            expanded = self._maybe_expand(reasoning)

        return {
            "has_reasoning": True,
            "folded": len(reasoning) > self.preview_chars,
            "expanded": expanded,
            "hint": hint,
            "full_reasoning": reasoning,
        }

    def _maybe_expand(self, reasoning: str) -> bool:
        """Prompt user once for on-demand full reasoning expansion."""
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
        """Write text to configured stream with immediate flush."""
        self.output_stream.write(text)
        self.output_stream.flush()
