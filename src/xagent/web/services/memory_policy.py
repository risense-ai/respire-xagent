"""Trusted host overrides for task memory eligibility."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MEMORY_POLICY_RESOLVER_FAILURE_REASON = "resolver_failed"


@dataclass(frozen=True)
class MemoryPolicyRequest:
    """Stable task fields supplied to the trusted host resolver."""

    task_id: int | None
    user_id: int | None
    agent_id: int | None
    source: str | None
    is_preview: bool


@dataclass(frozen=True)
class MemoryPolicyDecision:
    """Explicit memory eligibility and availability decision."""

    enabled: bool
    available: bool
    reason: str | None = None
    remote_status: str | None = None
    semantic_available: bool | None = None


MemoryPolicyResolver = Callable[[MemoryPolicyRequest], MemoryPolicyDecision]

_trusted_memory_policy_resolver: MemoryPolicyResolver | None = None


def set_trusted_memory_policy_resolver(
    resolver: MemoryPolicyResolver | None,
) -> None:
    """Install or clear the process-wide resolver owned by the trusted host."""

    if resolver is not None and not callable(resolver):
        raise TypeError("memory policy resolver must be callable")
    global _trusted_memory_policy_resolver
    _trusted_memory_policy_resolver = resolver


def resolve_trusted_memory_policy(
    request: MemoryPolicyRequest,
) -> MemoryPolicyDecision | None:
    """Resolve a host override, failing closed on errors or invalid decisions."""

    resolver = _trusted_memory_policy_resolver
    if resolver is None:
        return None

    try:
        decision = resolver(request)
        _validate_decision(decision)
        return decision
    except Exception:
        logger.exception("Trusted memory policy resolver failed")
        return MemoryPolicyDecision(
            enabled=False,
            available=False,
            reason=MEMORY_POLICY_RESOLVER_FAILURE_REASON,
        )


def _validate_decision(decision: object) -> None:
    if not isinstance(decision, MemoryPolicyDecision):
        raise TypeError("memory policy resolver returned an invalid decision type")
    if type(decision.enabled) is not bool or type(decision.available) is not bool:
        raise TypeError("memory policy decision flags must be bool")
    if decision.remote_status is not None and decision.remote_status not in {
        "ready",
        "degraded",
        "unavailable",
        "protocol_error",
    }:
        raise TypeError("invalid remote memory status")
    if (
        decision.semantic_available is not None
        and type(decision.semantic_available) is not bool
    ):
        raise TypeError("semantic availability must be bool")
    if decision.remote_status is not None:
        usable = decision.remote_status in {"ready", "degraded"}
        semantic = decision.remote_status == "ready"
        if (
            decision.available is not usable
            or decision.semantic_available is not semantic
        ):
            raise ValueError("remote memory state contradicts declared capabilities")
        if decision.remote_status == "degraded" and decision.reason is None:
            raise ValueError("degraded memory requires a reason")
    if decision.enabled and not decision.available:
        raise ValueError("unavailable memory cannot be enabled")
    if decision.reason is not None and (
        not isinstance(decision.reason, str) or not decision.reason.strip()
    ):
        raise TypeError("memory policy decision reason must be a non-empty string")
    if not decision.available and decision.reason is None:
        raise ValueError("unavailable memory requires a reason")
