#!/usr/bin/env python3
"""
v02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."
"""

import os
import sys
import json
import argparse
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.getcwd())

load_dotenv(override = True)

SYSTEM_PROMPT_PATH = "prompts/v01-v02.md"

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
- Write: echo 'content' > file, sed -i 's/old/new/g' file""",
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