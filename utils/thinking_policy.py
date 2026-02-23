"""Thinking capability detection and request parameter selection."""

from dataclasses import dataclass
from typing import Any, Dict


_ALLOWED_CAPABILITIES = {"auto", "toggle", "always", "never"}
_ALLOWED_PARAM_STYLES = {"auto", "enable_thinking", "reasoning_effort", "both"}
_ALLOWED_MODES = {"auto", "on", "off"}
_ALLOWED_REASONING_EFFORT = {"none", "low", "medium", "high"}


@dataclass
class ThinkingPolicyState:
    """Resolved thinking capability and parameter style for a runtime."""

    capability: str = "never"
    param_style: str = "none"


def resolve_thinking_policy(
    client: Any,
    model: str,
    capability_setting: str = "auto",
    param_style_setting: str = "auto",
) -> ThinkingPolicyState:
    """Resolve effective thinking support by manual setting or lightweight probes."""
    capability = (capability_setting or "auto").strip().lower()
    param_style = (param_style_setting or "auto").strip().lower()

    if capability not in _ALLOWED_CAPABILITIES:
        capability = "auto"
    if param_style not in _ALLOWED_PARAM_STYLES:
        param_style = "auto"

    if capability != "auto":
        resolved_style = param_style if param_style != "auto" else "enable_thinking"
        return ThinkingPolicyState(
            capability = capability,
            param_style = resolved_style,
        )

    if client is None or not model:
        return ThinkingPolicyState(capability = "never", param_style = "none")

    styles_to_try = [
        param_style,
    ] if param_style != "auto" else [
        "enable_thinking",
        "reasoning_effort",
        "both",
    ]

    for style in styles_to_try:
        supports_on = _probe_support(
            client = client,
            model = model,
            params = _params_for_enabled_state(style = style, enabled = True),
        )
        supports_off = _probe_support(
            client = client,
            model = model,
            params = _params_for_enabled_state(style = style, enabled = False),
        )

        if supports_on and supports_off:
            return ThinkingPolicyState(capability = "toggle", param_style = style)
        if supports_on:
            return ThinkingPolicyState(capability = "always", param_style = style)
        if supports_off:
            return ThinkingPolicyState(capability = "never", param_style = style)

    return ThinkingPolicyState(capability = "never", param_style = "none")


def build_thinking_params(
    policy: ThinkingPolicyState,
    thinking_mode: str,
    reasoning_effort: str,
) -> Dict[str, Any]:
    """Build request parameters from policy + user mode."""
    mode = (thinking_mode or "auto").strip().lower()
    effort = (reasoning_effort or "none").strip().lower()

    if mode not in _ALLOWED_MODES:
        mode = "auto"
    if effort not in _ALLOWED_REASONING_EFFORT:
        effort = "none"

    if policy.capability == "never" or policy.param_style == "none":
        return {}

    enabled = _resolve_enabled_state(policy = policy, mode = mode)
    if enabled is None:
        return {}

    if enabled:
        return _params_for_enabled_state(
            style = policy.param_style,
            enabled = True,
            reasoning_effort = effort,
        )

    return _params_for_enabled_state(style = policy.param_style, enabled = False)


def _resolve_enabled_state(policy: ThinkingPolicyState, mode: str):
    """Resolve whether request should force thinking on/off, or skip parameter."""
    if mode == "on":
        if policy.capability in {"toggle", "always"}:
            return True
        return None

    if mode == "off":
        if policy.capability == "toggle":
            return False
        return None

    # auto mode: only set explicit state when model is always-on.
    if policy.capability == "always":
        return True

    return None


def _params_for_enabled_state(
    style: str,
    enabled: bool,
    reasoning_effort: str = "low",
) -> Dict[str, Any]:
    """Translate desired enabled state to provider request params."""
    if style == "enable_thinking":
        return {"enable_thinking": enabled}

    if style == "reasoning_effort":
        effort = reasoning_effort if enabled else "none"
        return {"reasoning_effort": effort}

    if style == "both":
        effort = reasoning_effort if enabled else "none"
        return {
            "enable_thinking": enabled,
            "reasoning_effort": effort,
        }

    return {}


def _probe_support(client: Any, model: str, params: Dict[str, Any]) -> bool:
    """Run lightweight capability probe call."""
    try:
        client.chat.completions.create(
            model = model,
            messages = [{"role": "user", "content": "ping"}],
            max_tokens = 1,
            **params,
        )
        return True
    except Exception:
        return False
