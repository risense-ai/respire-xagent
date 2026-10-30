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

from typing import Any

import pytest

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
