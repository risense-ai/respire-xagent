"""Backend selection for the memory SaaS adapter (方案 §3.3, 裁决⑨).

Three wiring points are pinned here:

* ``memory_lifecycle.remote_admission_result`` -- the admission result the
  remote backend publishes, including the fail-closed answer for an
  incomplete configuration;
* ``DynamicMemoryStoreManager`` -- selecting the remote backend must publish
  a ``RemoteMemoryStore`` without ever reading the local embedding authority
  (remote mode has no authority row), while the unselected path is untouched;
* the T14 §B1-3 memory policy -- importing ``memory_lifecycle`` installs the
  trusted resolver, and that resolver opens memory for non-preview tasks on
  the remote backend only, gated on the service's readiness answer.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from xagent.core.memory import remote as remote_module
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.memory.remote import (
    READINESS_DEGRADED_REASON,
    READINESS_UNAVAILABLE_REASON,
    RemoteMemoryReadiness,
    RemoteMemoryStore,
)
from xagent.web import dynamic_memory_store as manager_module
from xagent.web import memory_lifecycle as lifecycle_module
from xagent.web.dynamic_memory_store import DynamicMemoryStoreManager
from xagent.web.memory_lifecycle import (
    MemoryLifecycleState,
    MemoryUnavailableError,
    remote_admission_result,
)
from xagent.web.revocable_memory_store import RevocableMemoryStore, unwrap_memory_store
from xagent.web.services import memory_policy as memory_policy_module
from xagent.web.services.memory_policy import (
    MemoryPolicyDecision,
    MemoryPolicyRequest,
    resolve_trusted_memory_policy,
)

#: Captured while this module is imported (collection), before any test
#: fixture runs: ``memory_lifecycle`` must have installed its resolver on
#: import, because the hook is process-wide and another suite's fixture may
#: clear it later.
_RESOLVER_AT_IMPORT = memory_policy_module._trusted_memory_policy_resolver  # noqa: SLF001


@pytest.fixture(autouse=True)
def trusted_user_context():
    from xagent.core.user_context import current_user_id

    token = current_user_id.set(23)
    try:
        yield
    finally:
        current_user_id.reset(token)


def _select_remote(monkeypatch: Any, *, key: str | None = "svc-key") -> None:
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://memsrv:8080")
    if key is None:
        monkeypatch.delenv("XAGENT_MEMORY_SERVICE_KEY", raising=False)
    else:
        monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", key)


def test_remote_admission_is_inactive_without_the_switch(monkeypatch: Any) -> None:
    monkeypatch.delenv("XAGENT_MEMORY_BACKEND", raising=False)

    assert remote_admission_result() is None


def test_remote_admission_publishes_a_ready_vector_store(monkeypatch: Any) -> None:
    _select_remote(monkeypatch)

    result = remote_admission_result()

    assert result is not None
    assert isinstance(result.store, RemoteMemoryStore)
    assert result.status.state is MemoryLifecycleState.READY
    assert result.status.mode is None
    assert result.status.remote_status == "unknown"
    assert result.status.vector_search is False
    # No local vector space exists to fingerprint.
    assert result.vector_space_fingerprint is None


def test_remote_admission_fails_closed_on_an_incomplete_configuration(
    monkeypatch: Any,
) -> None:
    _select_remote(monkeypatch, key=None)

    result = remote_admission_result()

    assert result is not None
    assert result.store is None
    assert result.status.state is MemoryLifecycleState.RESTART_REQUIRED


def test_manager_publishes_remote_without_reading_the_authority(
    monkeypatch: Any,
) -> None:
    _select_remote(monkeypatch)

    def fail_read() -> None:
        raise AssertionError("authority must not be read in remote mode")

    monkeypatch.setattr(manager_module, "_read_authority_snapshot", fail_read)
    monkeypatch.setattr(
        manager_module,
        "remote_memory_readiness",
        lambda: RemoteMemoryReadiness("ready", semantic_available=True),
    )
    manager = DynamicMemoryStoreManager()

    manager.admit()

    store = manager.get_memory_store()
    assert isinstance(unwrap_memory_store(store), RemoteMemoryStore)
    status = manager.status()
    assert status.state is MemoryLifecycleState.READY
    assert status.vector_search is True
    info = manager.get_store_info()
    assert info["store_type"] == "RemoteMemoryStore"
    assert info["is_lancedb"] is False


def test_manager_stays_closed_with_an_incomplete_remote_configuration(
    monkeypatch: Any,
) -> None:
    _select_remote(monkeypatch, key=None)
    manager = DynamicMemoryStoreManager()

    manager.admit()

    assert manager.status().state is MemoryLifecycleState.RESTART_REQUIRED
    with pytest.raises(MemoryUnavailableError):
        manager.get_memory_store()


# ---------------------------------------------------------------------------
# T14 §B1-3: the memory policy the remote backend registers
# ---------------------------------------------------------------------------


def _policy_request(
    *, is_preview: bool = False, agent_id: int | None = 41
) -> MemoryPolicyRequest:
    return MemoryPolicyRequest(
        task_id=17,
        user_id=23,
        agent_id=agent_id,
        source="a2a",
        is_preview=is_preview,
    )


def test_importing_memory_lifecycle_installs_the_memory_policy_resolver() -> None:
    assert _RESOLVER_AT_IMPORT is lifecycle_module.resolve_remote_memory_policy


def test_registration_installs_the_resolver_in_the_shared_hook() -> None:
    lifecycle_module.register_remote_memory_policy()
    installed = memory_policy_module._trusted_memory_policy_resolver  # noqa: SLF001

    assert installed is lifecycle_module.resolve_remote_memory_policy


# ---------------------------------------------------------------------------
# 缺陷二: store-info must describe the *upstream*, not only the local
# lifecycle publication. A ready lifecycle state is overridden by the
# adapter's health signal; a fenced lifecycle state always wins.
# ---------------------------------------------------------------------------


def _health(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "last_success_at": None,
        "last_success_status": None,
        "last_failure_at": None,
        "last_failure_reason": None,
        "last_event_at": None,
        "last_event_kind": None,
        "last_status": None,
        "last_path": None,
        "last_error_code": None,
        "last_error_message": None,
        "consecutive_transport_failures": 0,
    }
    base.update(overrides)
    return base


def _remote_manager(monkeypatch: Any) -> DynamicMemoryStoreManager:
    _select_remote(monkeypatch)
    # R2: the store-info report now derives its lifecycle truth from the
    # readiness probe, not the status code. Answer the probe so the
    # observed-upstream-health fold below is what these tests exercise.
    monkeypatch.setattr(
        manager_module,
        "remote_memory_readiness",
        lambda **_: RemoteMemoryReadiness(status="ready", semantic_available=True),
    )
    manager = DynamicMemoryStoreManager()
    manager.admit()
    return manager


def test_store_info_is_ready_when_the_last_upstream_call_succeeded(
    monkeypatch: Any,
) -> None:
    manager = _remote_manager(monkeypatch)
    monkeypatch.setattr(
        remote_module,
        "_upstream_health",
        _health(
            last_event_at=time.time(),
            last_event_kind="answer",
            last_status=200,
            last_path="/v1/memory/list",
            last_success_at=time.time(),
        ),
    )

    info = manager.get_store_info()

    assert info["store_type"] == "RemoteMemoryStore"
    assert info["state"] == "ready"
    assert "上游最近一次调用成功" in info["detail"]


def test_store_info_degrades_after_an_upstream_5xx_answer(monkeypatch: Any) -> None:
    manager = _remote_manager(monkeypatch)
    monkeypatch.setattr(
        remote_module,
        "_upstream_health",
        _health(
            last_event_at=time.time(),
            last_event_kind="answer",
            last_status=503,
            last_path="/v1/memory/notes",
            last_error_code="resource_exhausted",
            last_success_at=time.time(),
        ),
    )

    info = manager.get_store_info()

    assert info["state"] == "degraded"
    assert "HTTP 503" in info["detail"]
    assert "code=resource_exhausted" in info["detail"]


def test_store_info_is_unavailable_when_the_upstream_never_answered(
    monkeypatch: Any,
) -> None:
    manager = _remote_manager(monkeypatch)
    monkeypatch.setattr(
        remote_module,
        "_upstream_health",
        _health(
            last_event_at=time.time(),
            last_event_kind="transport_error",
            last_failure_at=time.time(),
            last_failure_reason="连接失败（ConnectionError）",
            last_path="/v1/memory/list",
            consecutive_transport_failures=3,
        ),
    )

    info = manager.get_store_info()

    assert info["state"] == "unavailable"
    assert "不可达" in info["detail"]


def test_store_info_keeps_the_lifecycle_truth_when_memory_is_fenced(
    monkeypatch: Any,
) -> None:
    _select_remote(monkeypatch, key=None)
    manager = DynamicMemoryStoreManager()
    manager.admit()

    def must_not_probe() -> None:
        raise AssertionError("a fenced store must not consult the upstream")

    monkeypatch.setattr(manager_module, "remote_upstream_report", must_not_probe)
    info = manager.get_store_info()

    assert info["state"] == MemoryLifecycleState.RESTART_REQUIRED.value
    assert info["store_type"] is None


def test_agent_bound_tasks_get_memory_when_the_service_is_ready(
    monkeypatch: Any,
) -> None:
    _select_remote(monkeypatch)
    monkeypatch.setattr(
        lifecycle_module,
        "remote_memory_readiness",
        lambda **_kwargs: RemoteMemoryReadiness("ready", semantic_available=True),
    )
    lifecycle_module.register_remote_memory_policy()

    decision = resolve_trusted_memory_policy(_policy_request())

    assert decision == MemoryPolicyDecision(
        enabled=True, available=True, remote_status="ready", semantic_available=True
    )


def test_agent_bound_tasks_leave_the_native_rule_alone_without_the_remote_backend(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("XAGENT_MEMORY_BACKEND", raising=False)
    monkeypatch.setattr(
        lifecycle_module,
        "remote_memory_readiness",
        lambda **_kwargs: pytest.fail("the probe must not run off the remote backend"),
    )
    lifecycle_module.register_remote_memory_policy()

    decision = resolve_trusted_memory_policy(_policy_request())

    # The native rule: an agent-bound task has memory disabled and the
    # published store decides availability.
    assert decision == MemoryPolicyDecision(enabled=False, available=True)


def test_preview_tasks_keep_the_native_rule(monkeypatch: Any) -> None:
    _select_remote(monkeypatch)
    monkeypatch.setattr(
        lifecycle_module,
        "remote_memory_readiness",
        lambda **_kwargs: pytest.fail("the probe must not run for a preview task"),
    )
    lifecycle_module.register_remote_memory_policy()

    decision = resolve_trusted_memory_policy(_policy_request(is_preview=True))

    assert decision == MemoryPolicyDecision(enabled=False, available=True)


def test_ordinary_tasks_use_remote_health(monkeypatch: Any) -> None:
    _select_remote(monkeypatch)
    monkeypatch.setattr(
        lifecycle_module,
        "remote_memory_readiness",
        lambda **_kwargs: RemoteMemoryReadiness("ready", semantic_available=True),
    )
    lifecycle_module.register_remote_memory_policy()

    decision = resolve_trusted_memory_policy(_policy_request(agent_id=None))

    assert decision == MemoryPolicyDecision(
        enabled=True, available=True, remote_status="ready", semantic_available=True
    )


def test_policy_fails_closed_when_the_service_is_not_ready(monkeypatch: Any) -> None:
    _select_remote(monkeypatch)
    monkeypatch.setattr(
        lifecycle_module,
        "remote_memory_readiness",
        lambda **_kwargs: RemoteMemoryReadiness(
            "unavailable", reason=READINESS_UNAVAILABLE_REASON
        ),
    )
    lifecycle_module.register_remote_memory_policy()

    decision = resolve_trusted_memory_policy(_policy_request())

    assert decision == MemoryPolicyDecision(
        enabled=False,
        available=False,
        reason=READINESS_UNAVAILABLE_REASON,
        remote_status="unavailable",
        semantic_available=False,
    )


class _Generation:
    """Minimal publication source: a fixed, always-current generation."""

    def __init__(self, generation: int = 1) -> None:
        self._generation = generation

    def publication_generation(self) -> int:
        return self._generation


class _JudgedDelegate:
    """Delegate exposing only the optional judged write."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, bool]] = []

    def remember(self, note: Any, *, force: bool = False) -> Any:
        self.calls.append((note, force))
        return MemoryResponse(
            success=True,
            memory_id="judged-1",
            metadata={"action": "parent", "parent_id": "p1"},
        )


def test_revocable_store_forwards_the_judged_write() -> None:
    """The wrapper must not hide ``remember`` from ``store_memory``.

    Without the forward, every wrapped store falls back to the plain-add
    path and the service's merge / attach / create judgement never runs.
    """
    delegate = _JudgedDelegate()
    store = RevocableMemoryStore(delegate, _Generation(), 1)  # type: ignore[arg-type]
    note = MemoryNote(content="a durable fact")

    response = store.remember(note, force=True)

    assert response.metadata == {"action": "parent", "parent_id": "p1"}
    assert delegate.calls == [(note, True)]


def test_revocable_store_reports_an_unsupported_judged_write() -> None:
    """A delegate without the capability raises, so the caller can fall back."""
    store = RevocableMemoryStore(object(), _Generation(), 1)  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        store.remember(MemoryNote(content="a durable fact"))


def test_degraded_policy_keeps_storage_available_and_reports_semantics(
    monkeypatch: Any,
) -> None:
    _select_remote(monkeypatch)
    monkeypatch.setattr(
        lifecycle_module,
        "remote_memory_readiness",
        lambda: RemoteMemoryReadiness("degraded", reason=READINESS_DEGRADED_REASON),
    )
    decision = lifecycle_module.resolve_remote_memory_policy(_policy_request())
    assert decision.enabled and decision.available
    assert decision.remote_status == "degraded"
    assert decision.semantic_available is False
    assert decision.reason == READINESS_DEGRADED_REASON


def test_remote_manager_reports_current_capabilities_and_recovery(
    monkeypatch: Any,
) -> None:
    _select_remote(monkeypatch)
    health = RemoteMemoryReadiness("degraded", reason=READINESS_DEGRADED_REASON)
    monkeypatch.setattr(manager_module, "remote_memory_readiness", lambda: health)
    manager = DynamicMemoryStoreManager()
    manager.admit()
    original = manager.get_memory_store()
    assert manager.status().remote_status == "degraded"
    assert manager.get_store_info()["supports_vector_search"] is False
    health = RemoteMemoryReadiness("unavailable", reason=READINESS_UNAVAILABLE_REASON)
    assert manager.status().state is MemoryLifecycleState.RETRYABLE_UNAVAILABLE
    assert manager.get_store_info()["reason"] == READINESS_UNAVAILABLE_REASON
    health = RemoteMemoryReadiness("ready", semantic_available=True)
    assert manager.get_store_info()["supports_vector_search"] is True
    assert manager.get_memory_store() is original


def test_runtime_policy_metadata_preserves_remote_capabilities(
    monkeypatch: Any,
) -> None:
    from types import SimpleNamespace

    from xagent.core.memory.in_memory import InMemoryMemoryStore
    from xagent.web.services import agent_service_manager as runtime

    _select_remote(monkeypatch)
    monkeypatch.setattr(
        lifecycle_module,
        "remote_memory_readiness",
        lambda: RemoteMemoryReadiness("degraded", reason=READINESS_DEGRADED_REASON),
    )
    lifecycle_module.register_remote_memory_policy()
    monkeypatch.setattr(runtime, "get_memory_store", InMemoryMemoryStore)
    policy = runtime.resolve_agent_service_memory_policy(
        task=SimpleNamespace(
            id=17, user_id=23, agent_id=41, source="a2a", agent_config={}
        )
    )
    assert policy.memory_enabled and policy.memory_available
    assert policy.memory_availability_reason == READINESS_DEGRADED_REASON
    assert policy.execution_metadata() == {
        "memory_remote_status": "degraded",
        "memory_semantic_available": False,
    }


@pytest.mark.parametrize("identity", [None, 0, "", " "])
@pytest.mark.parametrize("agent_id", [None, 41])
def test_policy_missing_identity_does_not_probe(
    monkeypatch: Any, identity: Any, agent_id: Any
) -> None:
    from dataclasses import replace

    from xagent.core.memory.remote import IDENTITY_REQUIRED_REASON

    _select_remote(monkeypatch)
    monkeypatch.setattr(
        lifecycle_module,
        "remote_memory_readiness",
        lambda **_: pytest.fail("identity must precede readiness"),
    )
    decision = lifecycle_module.resolve_remote_memory_policy(
        replace(_policy_request(), user_id=identity, agent_id=agent_id)
    )
    assert decision == MemoryPolicyDecision(
        enabled=False, available=False, reason=IDENTITY_REQUIRED_REASON
    )


def test_remote_user_reports_require_identity_before_any_probe(
    monkeypatch: Any,
) -> None:
    from xagent.core.memory.remote import RemoteMemoryStoreIdentityError
    from xagent.core.user_context import current_user_id

    _select_remote(monkeypatch)
    manager = DynamicMemoryStoreManager()
    manager.admit()
    # Also intercept R2's structured probe when this PR is integrated with R2.
    monkeypatch.setattr(
        manager_module,
        "remote_memory_readiness",
        lambda: pytest.fail("probe preceded identity"),
        raising=False,
    )
    token = current_user_id.set(None)
    try:
        with pytest.raises(RemoteMemoryStoreIdentityError):
            manager.status()
        with pytest.raises(RemoteMemoryStoreIdentityError):
            manager.get_store_info()
    finally:
        current_user_id.reset(token)


@pytest.fixture
def bound_remote_reports(monkeypatch: pytest.MonkeyPatch) -> Any:
    import hashlib

    from xagent.core.memory import agent_seats

    _select_remote(monkeypatch)
    registry = agent_seats.TrustedAgentSeatRegistry()
    registry.publish(
        service_url="http://memsrv:8080",
        user_id=23,
        runtime_profile=hashlib.sha256(b"svc-key").hexdigest(),
        agent_ids=[7],
        ttl_seconds=30,
        status="active",
    )
    monkeypatch.setattr(agent_seats, "_provider", registry)
    monkeypatch.setattr(agent_seats, "_provider_generation", 1)
    calls: list[str] = []

    def readiness() -> RemoteMemoryReadiness:
        calls.append("readyz")
        return RemoteMemoryReadiness("ready", semantic_available=True)

    def upstream() -> tuple[str, str]:
        calls.append("upstream")
        return "ready", "synthetic upstream report"

    monkeypatch.setattr(manager_module, "remote_memory_readiness", readiness)
    monkeypatch.setattr(manager_module, "remote_upstream_report", upstream)
    manager = DynamicMemoryStoreManager()
    manager.admit()
    return registry, manager, calls


@pytest.mark.parametrize("report", ["status", "get_store_info"])
@pytest.mark.parametrize(
    ("independent", "change"),
    [
        (independent, change)
        for independent in (False, True)
        for change in (
            "user",
            "missing",
            "zero_user",
            "empty_user",
            "blank_user",
            "mode",
            "endpoint",
            "key",
            "provider",
        )
        if independent or change != "provider"
    ],
)
def test_user_health_reports_cannot_revive_invalidated_scope(
    bound_remote_reports: Any,
    monkeypatch: pytest.MonkeyPatch,
    report: str,
    independent: bool,
    change: str,
) -> None:
    from xagent.core.memory.agent_seats import set_trusted_agent_seat_provider
    from xagent.core.memory.remote import (
        RemoteMemoryStoreIdentityError,
        delegated_agent_identity,
    )
    from xagent.core.user_context import current_user_id

    registry, manager, calls = bound_remote_reports
    initial_mode = "1" if independent else "0"
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", initial_mode)
    with delegated_agent_identity(7, store=manager.get_memory_store()):
        result = getattr(manager, report)()
        if report == "status":
            assert result.ready and result.vector_search
        else:
            assert result["state"] == "ready" and result["supports_vector_search"]
        answered = list(calls)
        if change in {"user", "missing", "zero_user", "empty_user", "blank_user"}:
            invalid_user: Any = {
                "user": 32,
                "missing": None,
                "zero_user": 0,
                "empty_user": "",
                "blank_user": " ",
            }[change]
            current_user_id.set(invalid_user)
        elif change == "mode":
            monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "0" if independent else "1")
        elif change == "endpoint":
            monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://other-service")
        elif change == "key":
            monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "other-runtime-key")
        else:
            set_trusted_agent_seat_provider(None)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            getattr(manager, report)()
        assert calls == answered
        current_user_id.set(23)
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", initial_mode)
        monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://memsrv:8080")
        monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "svc-key")
        if change == "provider":
            set_trusted_agent_seat_provider(registry)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            getattr(manager, report)()
        assert calls == answered
        with delegated_agent_identity(7, store=manager.get_memory_store()):
            getattr(manager, report)()
        after_fresh_scope = list(calls)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            getattr(manager, report)()
        assert calls == after_fresh_scope


@pytest.mark.parametrize("independent", [False, True])
def test_scope_change_during_readiness_blocks_upstream_report_probe(
    bound_remote_reports: Any,
    monkeypatch: pytest.MonkeyPatch,
    independent: bool,
) -> None:
    from xagent.core.memory.remote import (
        RemoteMemoryStoreIdentityError,
        delegated_agent_identity,
    )

    _registry, manager, calls = bound_remote_reports
    initial_mode = "1" if independent else "0"
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", initial_mode)
    original = manager_module.remote_memory_readiness

    def changed_after_probe() -> RemoteMemoryReadiness:
        result = original()
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "0" if independent else "1")
        return result

    monkeypatch.setattr(manager_module, "remote_memory_readiness", changed_after_probe)
    with delegated_agent_identity(7, store=manager.get_memory_store()):
        with pytest.raises(RemoteMemoryStoreIdentityError):
            manager.get_store_info()
        assert calls == ["readyz"]
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", initial_mode)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            manager.get_store_info()
        assert calls == ["readyz"]


@pytest.mark.parametrize("independent", [False, True])
def test_explicit_control_plane_report_does_not_reauthorize_invalid_user_scope(
    bound_remote_reports: Any,
    monkeypatch: pytest.MonkeyPatch,
    independent: bool,
) -> None:
    from xagent.core.memory.remote import (
        RemoteMemoryStoreIdentityError,
        delegated_agent_identity,
    )
    from xagent.core.user_context import current_user_id

    _registry, manager, calls = bound_remote_reports
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "1" if independent else "0")
    with delegated_agent_identity(7, store=manager.get_memory_store()):
        current_user_id.set(None)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            manager.status()
        assert calls == []
        assert manager.status(control_plane=True).ready
        assert manager.get_store_info(control_plane=True)["state"] == "ready"
        answered = list(calls)
        current_user_id.set(23)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            manager.get_store_info()
        assert calls == answered
