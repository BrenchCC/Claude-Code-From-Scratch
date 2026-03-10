#!/usr/bin/env python3
"""Minimal tool-calling agent loop with improved CLI output."""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.getcwd())


load_dotenv(override = True)

SYSTEM_PROMPT_PATH = "prompts/v01_agent_loop.md"

with open(SYSTEM_PROMPT_PATH, "r") as f:
    system_prompt = f.read().strip()

system_prompt = system_prompt.format(workspace = os.getcwd())

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": """Execute shell command. Common patterns include:
- Read: cat/head/tail, grep/find/rg/ls, wc -l
- Write: echo 'content' > file, sed -i 's/old/new/g' file
- Subagent: python v1_bash_agent_demo/bash_agent.py 'task description' (spawns isolated agent, returns summary)""",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


client = OpenAI(
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL") or None,
)
model = os.getenv("LLM_MODEL", "gpt-4o-mini")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None.
    """
    parser = argparse.ArgumentParser(
        description = "Run a minimal tool-calling agent loop."
    )
    parser.add_argument(
        "--max-turns",
        type = int,
        default = 20,
        help = "Maximum turns per user query (default: 20)."
    )
    parser.add_argument(
        "--tool-preview-chars",
        type = int,
        default = 300,
        help = "How many tool output characters to preview in terminal."
    )
    parser.add_argument(
        "--thinking",
        choices = ["auto", "on", "off"],
        default = "auto",
        help = "Model thinking switch compatibility mode."
    )
    parser.add_argument(
        "--show-thinking",
        action = "store_true",
        help = "Print thinking/reasoning text if provider returns it."
    )
    return parser.parse_args()


def style_line(char: str, width: int = 72) -> str:
    """Build a repeated style separator.

    Args:
        char: Character used for line repetition.
        width: Target line width.
    """
    return char * width


def extract_text_content(content: Any) -> str:
    """Extract readable text from assistant message content.

    Args:
        content: Message content in str/list/provider-object shape.
    """
    if isinstance(content, str):
        return content.strip()

    text_parts: List[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                value = item.get("text")
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
                continue
            value = getattr(item, "text", None)
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
    return "\n".join(text_parts).strip()


def extract_reasoning_content(message: Any) -> str:
    """Extract optional model reasoning/thinking text.

    Args:
        message: Assistant message object from API response.
    """
    reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        return ""

    if isinstance(reasoning, str):
        return reasoning.strip()

    summary = getattr(reasoning, "summary", None)
    if isinstance(summary, str):
        return summary.strip()

    if isinstance(summary, list):
        parts: List[str] = []
        for item in summary:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()

    return ""


def build_chat_kwargs(
    messages: List[Dict[str, Any]],
    thinking_mode: str
) -> Dict[str, Any]:
    """Build model request payload with thinking compatibility toggles.

    Args:
        messages: Full conversation message list.
        thinking_mode: Thinking flag, one of auto/on/off.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": TOOL,
        "tool_choice": "auto",
        "max_tokens": 16384,
    }

    if thinking_mode == "auto":
        return kwargs

    # Compat: different providers expose different switches.
    # Send common styles in extra_body and fall back at runtime if needed.
    kwargs["extra_body"] = {
        "enable_thinking": thinking_mode == "on",
        "thinking": {"type": "enabled" if thinking_mode == "on" else "disabled"},
    }
    return kwargs


def create_chat_completion(
    messages: List[Dict[str, Any]],
    thinking_mode: str
) -> Any:
    """Create chat completion with lightweight provider-compat fallback.

    Args:
        messages: Full conversation message list.
        thinking_mode: Thinking flag, one of auto/on/off.
    """
    kwargs = build_chat_kwargs(messages = messages, thinking_mode = thinking_mode)
    try:
        return client.chat.completions.create(**kwargs)
    except Exception:
        # Some compatible providers reject unknown thinking fields.
        kwargs.pop("extra_body", None)
        return client.chat.completions.create(**kwargs)


def parse_tool_arguments(raw_arguments: str) -> Dict[str, Any]:
    """Parse tool-call argument string safely.

    Args:
        raw_arguments: Raw JSON arguments from tool call.
    """
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return parsed
    return {}


def render_assistant_message(message: Any, show_thinking: bool) -> None:
    """Render assistant output to terminal.

    Args:
        message: Assistant message object.
        show_thinking: Whether to print reasoning/thinking text.
    """
    text = extract_text_content(getattr(message, "content", None))
    if text:
        print(f"\033[32m{style_line('-')}\033[0m")
        print("\033[32massistant>\033[0m")
        print(text)

    if show_thinking:
        reasoning_text = extract_reasoning_content(message)
        if reasoning_text:
            print(f"\033[35m{style_line('-')}\033[0m")
            print("\033[35mthinking>\033[0m")
            print(reasoning_text)


def run_bash(command: str) -> str:
    """Run shell command safely.

    Args:
        command: Shell command string from tool call.
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell = True,
            cwd = os.getcwd(),
            capture_output = True,
            text = True,
            timeout = 120
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def agent_loop(
    messages: List[Dict[str, Any]],
    max_turns: int = 20,
    tool_preview_chars: int = 300,
    thinking_mode: str = "auto",
    show_thinking: bool = False
) -> Optional[Any]:
    """Run the assistant-tool loop.

    Args:
        messages: Chat message list.
        max_turns: Maximum loop turns.
        tool_preview_chars: Tool output preview length for terminal.
        thinking_mode: Thinking mode switch for model request.
        show_thinking: Whether to print thinking/reasoning output.
    """
    turn = 0
    last_message: Optional[Any] = None

    while turn < max_turns:
        turn += 1
        print(f"\033[36m{style_line('=')}\033[0m")
        print(f"\033[36mturn {turn}/{max_turns}\033[0m")

        response = create_chat_completion(
            messages = messages,
            thinking_mode = thinking_mode,
        )
        message = response.choices[0].message
        messages.append(message)
        last_message = message

        render_assistant_message(message = message, show_thinking = show_thinking)

        # If model returns no tool call, finish.
        if not message.tool_calls:
            return last_message

        print(f"\033[33mtool calls: {len(message.tool_calls)}\033[0m")
        results = []

        # Execute each tool call, collect results
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = parse_tool_arguments(tool_call.function.arguments)

            if function_name == "bash":
                command = arguments.get("command", "")
                if not command:
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Error: Missing 'command' in tool arguments",
                    })
                    continue

                print(f"\033[33mtool> bash\033[0m")
                print(f"\033[33m$ {command}\033[0m")

                output = run_bash(command)
                preview = output[:tool_preview_chars]
                print(preview)
                if len(output) > len(preview):
                    print(
                        f"\033[90m... output truncated ({len(output)} chars total)\033[0m"
                    )

                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                })
            else:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: Unsupported tool {function_name}",
                })

        messages.extend(results)
    return last_message

if __name__ == "__main__":
    args = parse_args()
    history: List[Dict[str, Any]] = []

    print("\033[36m" + style_line("=") + "\033[0m")
    print("\033[36mv01 agent loop\033[0m")
    print(f"\033[36mmodel: Doubao-Seed-2.0\033[0m")
    print(f"\033[36mthinking mode: {args.thinking}\033[0m")
    print("\033[36mtype 'q' / 'exit' / empty line to quit\033[0m")
    print("\033[36m" + style_line("=") + "\033[0m")

    while True:
        try:
            query = input("\033[34muser > \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(
            messages = history,
            max_turns = args.max_turns,
            tool_preview_chars = args.tool_preview_chars,
            thinking_mode = args.thinking,
            show_thinking = args.show_thinking,
        )
        print()
