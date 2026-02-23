"""Unified chat completion wrapper for stream and non-stream modes."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


_THINKING_KEYS = {"enable_thinking", "reasoning_effort"}


@dataclass
class LLMCallResult:
    """Normalized result returned by the shared LLM call wrapper."""

    assistant_content: str
    assistant_reasoning: str
    tool_calls: List[Dict[str, Any]]
    raw_metadata: Dict[str, Any]


def call_chat_completion(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 8192,
    stream: bool = False,
    thinking_params: Optional[Dict[str, Any]] = None,
    on_content_chunk: Optional[Callable[[str], None]] = None,
    on_reasoning_chunk: Optional[Callable[[str], None]] = None,
) -> LLMCallResult:
    """Call chat completion once, with one-shot thinking-param fallback retry."""
    request = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if tools is not None:
        request["tools"] = tools

    thinking_params = thinking_params or {}
    request.update(thinking_params)

    if stream:
        request["stream"] = True

    try:
        return _invoke_once(
            client = client,
            request = request,
            stream = stream,
            on_content_chunk = on_content_chunk,
            on_reasoning_chunk = on_reasoning_chunk,
        )
    except Exception as exc:
        if not thinking_params or not _looks_like_thinking_param_error(exc):
            raise

        retry_request = {
            key: value
            for key, value in request.items()
            if key not in _THINKING_KEYS
        }
        result = _invoke_once(
            client = client,
            request = retry_request,
            stream = stream,
            on_content_chunk = on_content_chunk,
            on_reasoning_chunk = on_reasoning_chunk,
        )
        result.raw_metadata["thinking_params_stripped_retry"] = True
        result.raw_metadata["thinking_retry_error"] = str(exc)
        return result


def build_assistant_message(result: LLMCallResult) -> Dict[str, Any]:
    """Convert normalized result to OpenAI-compatible assistant message dict."""
    assistant_message = {
        "role": "assistant",
        "content": result.assistant_content or "",
    }
    if result.tool_calls:
        assistant_message["tool_calls"] = result.tool_calls
    return assistant_message


def _invoke_once(
    client: Any,
    request: Dict[str, Any],
    stream: bool,
    on_content_chunk: Optional[Callable[[str], None]],
    on_reasoning_chunk: Optional[Callable[[str], None]],
) -> LLMCallResult:
    """Single API call execution path."""
    if stream:
        return _invoke_stream(
            client = client,
            request = request,
            on_content_chunk = on_content_chunk,
            on_reasoning_chunk = on_reasoning_chunk,
        )

    response = client.chat.completions.create(**request)
    message = response.choices[0].message
    content = _extract_content_from_message(message)
    reasoning = _extract_reasoning_from_message(message)
    tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None))

    return LLMCallResult(
        assistant_content = content,
        assistant_reasoning = reasoning,
        tool_calls = tool_calls,
        raw_metadata = {
            "stream": False,
            "response": _safe_model_dump(response),
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "usage": _safe_model_dump(getattr(response, "usage", None)),
        },
    )


def _invoke_stream(
    client: Any,
    request: Dict[str, Any],
    on_content_chunk: Optional[Callable[[str], None]],
    on_reasoning_chunk: Optional[Callable[[str], None]],
) -> LLMCallResult:
    """Streaming API path with tool-call assembly and optional callbacks."""
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_buffers: Dict[int, Dict[str, Any]] = {}

    last_id = None
    last_model = None
    chunk_count = 0

    stream_iter = client.chat.completions.create(**request)
    for chunk in stream_iter:
        chunk_count += 1
        last_id = getattr(chunk, "id", last_id)
        last_model = getattr(chunk, "model", last_model)

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        content_piece = _extract_content_from_delta(delta)
        if content_piece:
            content_parts.append(content_piece)
            if on_content_chunk:
                on_content_chunk(content_piece)

        reasoning_piece = _extract_reasoning_from_delta(delta)
        if reasoning_piece:
            reasoning_parts.append(reasoning_piece)
            if on_reasoning_chunk:
                on_reasoning_chunk(reasoning_piece)

        delta_tool_calls = getattr(delta, "tool_calls", None)
        if delta_tool_calls:
            _merge_stream_tool_calls(tool_buffers = tool_buffers, delta_tool_calls = delta_tool_calls)

    tool_calls = [tool_buffers[index] for index in sorted(tool_buffers.keys())]

    return LLMCallResult(
        assistant_content = "".join(content_parts),
        assistant_reasoning = "".join(reasoning_parts),
        tool_calls = tool_calls,
        raw_metadata = {
            "stream": True,
            "chunk_count": chunk_count,
            "response_id": last_id,
            "model": last_model,
        },
    )


def _extract_content_from_message(message: Any) -> str:
    """Extract assistant content text from non-stream message payload."""
    return _coerce_text(getattr(message, "content", ""))


def _extract_reasoning_from_message(message: Any) -> str:
    """Extract reasoning text from non-stream message payload."""
    segments = []
    for attr_name in ["reasoning", "reasoning_content", "thinking"]:
        raw = getattr(message, attr_name, None)
        if raw:
            segments.append(_coerce_text(raw))

    # Some providers embed typed blocks inside content arrays.
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for part in content:
            part_type = _read_obj(part, "type")
            if part_type in {"reasoning", "thinking"}:
                segments.append(_coerce_text(_read_obj(part, "text") or _read_obj(part, "content")))

    return "".join(segment for segment in segments if segment)


def _extract_content_from_delta(delta: Any) -> str:
    """Extract streamed assistant content from delta payload."""
    content = getattr(delta, "content", None)
    if isinstance(content, list):
        segments = []
        for part in content:
            part_type = _read_obj(part, "type")
            if part_type in {"reasoning", "thinking"}:
                continue
            segments.append(_coerce_text(_read_obj(part, "text") or _read_obj(part, "content")))
        return "".join(segments)

    return _coerce_text(content)


def _extract_reasoning_from_delta(delta: Any) -> str:
    """Extract streamed reasoning text from delta payload."""
    segments = []
    for attr_name in ["reasoning", "reasoning_content", "thinking"]:
        raw = getattr(delta, attr_name, None)
        if raw:
            segments.append(_coerce_text(raw))

    content = getattr(delta, "content", None)
    if isinstance(content, list):
        for part in content:
            part_type = _read_obj(part, "type")
            if part_type in {"reasoning", "thinking"}:
                segments.append(_coerce_text(_read_obj(part, "text") or _read_obj(part, "content")))

    return "".join(segment for segment in segments if segment)


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize tool call objects to plain dicts."""
    normalized = []
    if not tool_calls:
        return normalized

    for tool_call in tool_calls:
        function_payload = _read_obj(tool_call, "function") or {}
        normalized.append(
            {
                "id": _read_obj(tool_call, "id"),
                "type": _read_obj(tool_call, "type") or "function",
                "function": {
                    "name": _read_obj(function_payload, "name") or "",
                    "arguments": _read_obj(function_payload, "arguments") or "{}",
                },
            }
        )
    return normalized


def _merge_stream_tool_calls(tool_buffers: Dict[int, Dict[str, Any]], delta_tool_calls: Any) -> None:
    """Merge incremental stream tool-call chunks by index."""
    for delta_tool_call in delta_tool_calls:
        raw_index = _read_obj(delta_tool_call, "index")
        index = int(raw_index) if raw_index is not None else len(tool_buffers)

        if index not in tool_buffers:
            tool_buffers[index] = {
                "id": _read_obj(delta_tool_call, "id") or f"call_{index}",
                "type": _read_obj(delta_tool_call, "type") or "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }

        buffer = tool_buffers[index]
        tool_id = _read_obj(delta_tool_call, "id")
        if tool_id:
            buffer["id"] = tool_id

        function_payload = _read_obj(delta_tool_call, "function")
        if function_payload:
            name_piece = _read_obj(function_payload, "name")
            if name_piece:
                existing_name = buffer["function"]["name"]
                if not existing_name:
                    buffer["function"]["name"] = name_piece
                elif not existing_name.endswith(name_piece):
                    buffer["function"]["name"] += name_piece

            args_piece = _read_obj(function_payload, "arguments")
            if args_piece:
                buffer["function"]["arguments"] += args_piece


def _coerce_text(value: Any) -> str:
    """Flatten value to text conservatively."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(_coerce_text(item) for item in value)

    if isinstance(value, dict):
        if "text" in value:
            return _coerce_text(value.get("text"))
        if "content" in value:
            return _coerce_text(value.get("content"))
        if "reasoning" in value:
            return _coerce_text(value.get("reasoning"))
        return ""

    for attr_name in ["text", "content", "reasoning"]:
        attr_value = getattr(value, attr_name, None)
        if attr_value is not None:
            return _coerce_text(attr_value)

    return str(value)


def _read_obj(obj: Any, key: str) -> Any:
    """Read key from object or dict safely."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _safe_model_dump(obj: Any) -> Any:
    """Best-effort conversion of SDK objects to plain dicts."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            return str(obj)

    return str(obj)


def _looks_like_thinking_param_error(exc: Exception) -> bool:
    """Check if an exception likely comes from unsupported thinking params."""
    text = str(exc).lower()
    signals = [
        "enable_thinking",
        "reasoning_effort",
        "unknown parameter",
        "unexpected keyword",
        "extra_forbidden",
        "not permitted",
    ]
    return any(signal in text for signal in signals)
