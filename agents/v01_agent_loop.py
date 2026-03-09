#!/usr/bin/env python3
"""
v01_agent_loop.py - Minimal tool-calling agent loop demo.

Core pattern:
1) Ask model with tools schema.
2) If assistant returns tool_calls, execute each tool.
3) Append tool results back to messages.
4) Continue until assistant stops calling tools.
"""

#!/usr/bin/env python3
"""
v01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""
import os
import sys
import json
import argparse
import subprocess
from typing import Any, Dict, List

from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.getcwd())


load_dotenv(override = True)

client = OpenAI(
        api_key = os.getenv("LLM_API_KEY"),
        base_url = os.getenv("LLM_BASE_URL") or None,
    )
model = os.getenv("LLM_MODEL", "gpt-4o-mini")

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


