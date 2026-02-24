"""Thinking capability detection and request parameter selection."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set


_ALLOWED_CAPABILITIES = {"auto", "toggle", "always", "never"}
_ALLOWED_PARAM_STYLES = {"auto", "enable_thinking", "reasoning_effort", "both"}
_ALLOWED_MODES = {"auto", "on", "off"}
_ALLOWED_REASONING_EFFORT = {"none", "low", "medium", "high"}


@dataclass
class ThinkingPolicyState:
    """Resolved thinking capability and parameter style for a runtime.

    Args:
        capability: Effective capability flag.
        param_style: Effective parameter style.
    """

    capability: str = "never"
    param_style: str = "none"


def resolve_thinking_policy(
    client: Any,
    model: str,
    capability_setting: str = "auto",
    param_style_setting: str = "auto",
) -> ThinkingPolicyState:
    """Resolve effective thinking support by manual setting or lightweight probes.

    Args:
        client: LLM client used for probe calls.
        model: Target model identifier.
        capability_setting: Manual capability override.
        param_style_setting: Manual thinking-param style override.
    """
    capability = _normalize_setting(
        value = capability_setting,
        allowed = _ALLOWED_CAPABILITIES,
        default = "auto",
    )
    param_style = _normalize_setting(
        value = param_style_setting,
        allowed = _ALLOWED_PARAM_STYLES,
        default = "auto",
    )

    if capability != "auto":
        resolved_style = param_style if param_style != "auto" else "enable_thinking"
        return ThinkingPolicyState(
            capability = capability,
            param_style = resolved_style,
        )

    if client is None or not model:
        return ThinkingPolicyState(capability = "never", param_style = "none")

    for style in _styles_to_probe(param_style = param_style):
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
    """Build request parameters from policy + user mode.

    Args:
        policy: Resolved runtime policy.
        thinking_mode: Requested mode (auto/on/off).
        reasoning_effort: Requested effort (none/low/medium/high).
    """
    mode = _normalize_setting(
        value = thinking_mode,
        allowed = _ALLOWED_MODES,
        default = "auto",
    )
    effort = _normalize_setting(
        value = reasoning_effort,
        allowed = _ALLOWED_REASONING_EFFORT,
        default = "none",
    )

    if policy.capability == "never" or policy.param_style == "none":
        return {}

    enabled = _resolve_enabled_state(policy = policy, mode = mode)
    if enabled is None:
        # No explicit request should be sent in this combination.
        return {}

    return _params_for_enabled_state(
        style = policy.param_style,
        enabled = enabled,
        reasoning_effort = effort,
    )


def _normalize_setting(value: Any, allowed: Set[str], default: str) -> str:
    """Normalize one string setting with allowed-set validation.

    Args:
        value: Raw user setting.
        allowed: Allowed normalized values.
        default: Fallback value.
    """
    normalized = (value or default).strip().lower()
    if normalized in allowed:
        return normalized
    return default


def _styles_to_probe(param_style: str) -> list:
    """Build probe style order.

    Args:
        param_style: User-selected style or auto.
    """
    if param_style != "auto":
        return [param_style]
    return ["enable_thinking", "reasoning_effort", "both"]


def _resolve_enabled_state(policy: ThinkingPolicyState, mode: str) -> Optional[bool]:
    """Resolve whether request should force thinking on/off, or skip parameter.

    Args:
        policy: Resolved runtime policy.
        mode: Requested mode (auto/on/off).
    """
    if mode == "on":
        if policy.capability in {"toggle", "always"}:
            return True
        return None

    if mode == "off":
        if policy.capability == "toggle":
            return False
        return None

    # In auto mode, only always-on capability sends an explicit enable signal.
    if policy.capability == "always":
        return True

    return None


def _params_for_enabled_state(
    style: str,
    enabled: bool,
    reasoning_effort: str = "low",
) -> Dict[str, Any]:
    """Translate desired enabled state to provider request params.

    Args:
        style: Param style identifier.
        enabled: Whether thinking should be enabled.
        reasoning_effort: Effort level used when enabled.
    """
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
    """Run lightweight capability probe call.

    Args:
        client: LLM client.
        model: Target model identifier.
        params: Probe parameters to test.
    """
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
