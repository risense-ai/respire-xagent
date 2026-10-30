"""Delegated memory wiring for ``AgentTool`` (T14 §B2).

``AgentTool.execute_delegated_runtime`` used to construct its child
``AgentService`` with ``memory_enabled=False`` unconditionally, so a delegated
run could never reach the delegating task's memory. §B2 replaces that with
``_resolve_delegated_parent_memory`` and three construction arguments.

These tests pin the resolution table and the wiring it feeds:

* remote backend and a ready service -> the child gets the *same* store object
  the delegating task runs on, memory enabled, and the memory preamble in its
  system prompt;
* no remote backend -> the historical behaviour, unchanged: no store, memory
  disabled, no preamble, and the readiness probe is never consulted;
* remote backend but an unreachable service -> memory disabled with the
  probe's reason, no preamble, and the delegation itself still completes;
* any failure while resolving -> the same fail-closed answer, never an
  exception out of the resolver.

The child's construction is observed through a recording ``AgentService``
double, so the assertions are about the arguments the runtime really passes,
not about a re-implementation of them.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
import requests

import xagent.core.tools.adapters.vibe.agent_tool as mod
from xagent.core.memory.remote import READINESS_UNAVAILABLE_REASON
from xagent.core.tools.adapters.vibe.agent_tool import (
    _DELEGATED_MEMORY_DISABLED_REASON,
    _DELEGATED_MEMORY_PREAMBLE,
    _DELEGATED_MEMORY_UNAVAILABLE_REASON,
    AgentTool,
    _resolve_delegated_parent_memory,
)


class _SucceedingConfig:
    def close(self) -> None:
        return None


class _DelegatedQuery:
    def __init__(self, agent: SimpleNamespace) -> None:
        self._agent = agent

    def filter(self, *_args: object) -> "_DelegatedQuery":
        return self

    def first(self) -> SimpleNamespace:
        return self._agent


class _DelegatedSession:
    def __init__(self, agent: SimpleNamespace) -> None:
        self._agent = agent

    def query(self, *_args: object) -> _DelegatedQuery:
        return _DelegatedQuery(self._agent)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


class _RecordingAgentService:
    """Stand-in for ``AgentService`` that keeps what the call site passed."""

    instances: list["_RecordingAgentService"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.contexts: list[dict[str, object] | None] = []
        self.workspace = None
        _RecordingAgentService.instances.append(self)

    async def execute_task(self, **kwargs: object) -> dict[str, str]:
        context = kwargs.get("context")
        assert context is None or isinstance(context, dict)
        self.contexts.append(context)
        return {"output": "child answer"}

    @property
    def system_prompt(self) -> object:
        context = self.contexts[-1] if self.contexts else None
        return (context or {}).get("system_prompt")


def _delegated_tool(*, agent_owner_id: int = 1) -> AgentTool:
    return AgentTool(
        agent_id=1,
        agent_name="Delegated",
        agent_description="d",
        session_factory=lambda: _DelegatedSession(
            SimpleNamespace(
                id=1,
                user_id=agent_owner_id,
                status="published",
                name="Delegated",
                instructions=None,
                knowledge_bases=None,
                skills=None,
                tool_categories=[],
                models={"general": 1},
                execution_mode=None,
            )
        ),
        user_id=1,
        tool_name="delegated",
        tool_description="d",
    )


@pytest.fixture(autouse=True)
def _reset_recorded_instances() -> None:
    _RecordingAgentService.instances = []


def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy collaborators, keeping the memory resolution real."""
    import xagent.core.agent.service as service_module
    import xagent.core.tools.adapters.vibe.agent_model_resolution as resolution

    monkeypatch.setattr(mod, "WebToolConfig", lambda **_kwargs: _SucceedingConfig())
    monkeypatch.setattr(service_module, "AgentService", _RecordingAgentService)
    monkeypatch.setattr(
        resolution,
        "resolve_agent_model_llms",
        lambda *_args: (object(), None, None, None),
    )


def _select_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://memsrv:8080")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "svc-key")
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "0")


def _patch_readiness(
    monkeypatch: pytest.MonkeyPatch, answer: tuple[bool, str | None]
) -> None:
    from xagent.core.memory import remote as remote_module

    monkeypatch.setattr(remote_module, "remote_memory_ready", lambda **_k: answer)


def _patch_published_store(monkeypatch: pytest.MonkeyPatch, store: object) -> None:
    from xagent.web import dynamic_memory_store as store_module

    monkeypatch.setattr(store_module, "get_memory_store", lambda: store)


def _patch_store_lookup(monkeypatch: pytest.MonkeyPatch, lookup: object) -> None:
    """Install an arbitrary ``get_memory_store`` replacement."""
    from xagent.web import dynamic_memory_store as store_module

    monkeypatch.setattr(store_module, "get_memory_store", lookup)


def _forbidden_store(message: str) -> object:
    """A store lookup that must never run, so the reason is reported, not eaten."""

    def refuse() -> object:
        pytest.fail(message)

    return refuse


async def _run_delegation(
    monkeypatch: pytest.MonkeyPatch, *, agent_owner_id: int = 1
) -> object:
    _patch_runtime(monkeypatch)
    tool = _delegated_tool(agent_owner_id=agent_owner_id)
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_k: None)
    return await tool.run_json_async({"task": "run"})


# ---------------------------------------------------------------------------
# The resolution table
# ---------------------------------------------------------------------------


def test_no_remote_backend_disables_delegated_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XAGENT_MEMORY_BACKEND", raising=False)
    _patch_readiness(
        monkeypatch, (True, None)
    )  # a probe here would be the bug, not the answer

    assert _resolve_delegated_parent_memory(user_id=1) == (
        None,
        False,
        _DELEGATED_MEMORY_DISABLED_REASON,
    )


def test_ready_service_publishes_the_parent_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_remote(monkeypatch)
    _patch_readiness(monkeypatch, (True, None))
    parent_store = object()
    _patch_published_store(monkeypatch, parent_store)

    assert _resolve_delegated_parent_memory(user_id=1) == (parent_store, True, None)


def test_unready_service_disables_memory_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_remote(monkeypatch)
    _patch_readiness(monkeypatch, (False, READINESS_UNAVAILABLE_REASON))
    _patch_store_lookup(
        monkeypatch,
        _forbidden_store("an unreachable service must not hand out a store"),
    )

    assert _resolve_delegated_parent_memory(user_id=1) == (
        None,
        False,
        READINESS_UNAVAILABLE_REASON,
    )


def test_unready_service_without_a_reason_reports_the_generic_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_remote(monkeypatch)
    _patch_readiness(monkeypatch, (False, None))
    _patch_store_lookup(
        monkeypatch,
        _forbidden_store("an unreachable service must not hand out a store"),
    )

    assert _resolve_delegated_parent_memory(user_id=1) == (
        None,
        False,
        _DELEGATED_MEMORY_UNAVAILABLE_REASON,
    )


def test_a_broken_store_lookup_fails_closed_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _select_remote(monkeypatch)
    _patch_readiness(monkeypatch, (True, None))

    def explode() -> object:
        raise RuntimeError("publication sentinel")

    from xagent.web import dynamic_memory_store as store_module

    monkeypatch.setattr(store_module, "get_memory_store", explode)

    assert _resolve_delegated_parent_memory(user_id=1) == (
        None,
        False,
        _DELEGATED_MEMORY_UNAVAILABLE_REASON,
    )
    assert "Delegated memory could not be resolved" in caplog.text


# ---------------------------------------------------------------------------
# What the child AgentService is constructed with
# ---------------------------------------------------------------------------


async def test_child_receives_the_parent_store_and_the_memory_preamble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_remote(monkeypatch)
    _patch_readiness(monkeypatch, (True, None))
    parent_store = object()
    _patch_published_store(monkeypatch, parent_store)

    result = await _run_delegation(monkeypatch)

    assert result == {"response": "child answer"}
    (child,) = _RecordingAgentService.instances
    # The very object the delegating task runs on, not a copy and not a
    # throwaway store: what the child writes must outlive the run.
    assert child.kwargs["memory"] is parent_store
    assert child.kwargs["memory_enabled"] is True
    assert child.kwargs["memory_available"] is True
    assert child.kwargs["memory_availability_reason"] is None
    assert child.kwargs["user_interaction_enabled"] is False
    assert child.system_prompt == _DELEGATED_MEMORY_PREAMBLE


async def test_the_preamble_follows_the_agent_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_remote(monkeypatch)
    _patch_readiness(monkeypatch, (True, None))
    _patch_published_store(monkeypatch, object())

    def session_factory() -> _DelegatedSession:
        return _DelegatedSession(
            SimpleNamespace(
                id=1,
                name="Delegated",
                instructions="Answer only in bullet points.",
                knowledge_bases=None,
                skills=None,
                tool_categories=[],
                models={"general": 1},
                execution_mode=None,
            )
        )

    _patch_runtime(monkeypatch)
    tool = _delegated_tool()
    tool._session_factory = session_factory
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_k: None)

    await tool.run_json_async({"task": "run"})

    (child,) = _RecordingAgentService.instances
    assert child.system_prompt == (
        f"Answer only in bullet points.\n\n{_DELEGATED_MEMORY_PREAMBLE}"
    )


async def test_no_remote_backend_keeps_the_historical_child_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XAGENT_MEMORY_BACKEND", raising=False)
    _patch_readiness(
        monkeypatch, (True, None)
    )  # a probe here would be the bug, not the answer
    _patch_store_lookup(
        monkeypatch, _forbidden_store("a local backend must not publish a remote store")
    )

    result = await _run_delegation(monkeypatch)

    assert result == {"response": "child answer"}
    (child,) = _RecordingAgentService.instances
    assert child.kwargs["memory"] is None
    assert child.kwargs["memory_enabled"] is False
    assert child.kwargs["memory_available"] is False
    assert child.kwargs["memory_availability_reason"] == (
        _DELEGATED_MEMORY_DISABLED_REASON
    )
    # The prompt must not describe a capability the runtime withheld.
    assert child.system_prompt is None


async def test_an_unreachable_service_degrades_without_failing_the_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_remote(monkeypatch)
    _patch_readiness(monkeypatch, (False, READINESS_UNAVAILABLE_REASON))
    _patch_store_lookup(
        monkeypatch,
        _forbidden_store("an unreachable service must not hand out a store"),
    )

    result = await _run_delegation(monkeypatch)

    # Memory being unavailable is a degraded feature, not a delegation failure.
    assert result == {"response": "child answer"}
    (child,) = _RecordingAgentService.instances
    assert child.kwargs["memory"] is None
    assert child.kwargs["memory_enabled"] is False
    assert child.kwargs["memory_available"] is False
    assert child.kwargs["memory_availability_reason"] == (READINESS_UNAVAILABLE_REASON)
    assert child.system_prompt is None


# ---------------------------------------------------------------------------
# T14 §0.2: the delegated child's own agent seat
# ---------------------------------------------------------------------------


class _FakeStoreSession:
    """Replays scripted answers and records the remote store's requests."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def request(self, method: str, url: str, **kwargs: object) -> requests.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return cast(requests.Response, item)


def _json_response(payload: object) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode("utf-8")  # noqa: SLF001
    response.headers["Content-Type"] = "application/json"
    return response


class _MemoryCallingAgentService(_RecordingAgentService):
    """Runs one memory call inside the child's execution context."""

    async def execute_task(self, **kwargs: object) -> dict[str, str]:
        memory = self.kwargs.get("memory")
        if memory is not None:
            memory.search("child memory call")  # type: ignore[attr-defined]
        return await super().execute_task(**kwargs)


def _patch_runtime_with_service(
    monkeypatch: pytest.MonkeyPatch, service_cls: type
) -> None:
    import xagent.core.agent.service as service_module

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(service_module, "AgentService", service_cls)


def _publish_remote_store(monkeypatch: pytest.MonkeyPatch, session: object) -> object:
    from xagent.core.memory.remote import RemoteMemoryStore

    store = RemoteMemoryStore(
        "http://memsrv:8080",
        "svc-key",
        session=session,  # type: ignore[arg-type]
    )
    _patch_published_store(monkeypatch, store)
    return store


def _patch_seats(monkeypatch: pytest.MonkeyPatch, answer: object) -> None:
    from xagent.core.memory import agent_seats

    registry = agent_seats.TrustedAgentSeatRegistry()
    if callable(answer):

        def provider(_request):
            return answer()
    else:
        refs, _reason = answer
        if refs is not None:
            registry.publish(
                service_url="http://memsrv:8080",
                user_id=1,
                runtime_profile=hashlib.sha256(b"svc-key").hexdigest(),
                agent_ids=[int(ref) for ref in refs],
                ttl_seconds=30,
                status="active",
            )
        provider = registry
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "1")
    monkeypatch.setattr(agent_seats, "_provider", provider)
    monkeypatch.setattr(agent_seats, "_provider_generation", 1)


async def _run_delegation_with_memory_call(
    monkeypatch: pytest.MonkeyPatch, session: _FakeStoreSession
) -> dict[str, object]:
    _select_remote(monkeypatch)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "1")
    _patch_readiness(monkeypatch, (True, None))
    _publish_remote_store(monkeypatch, session)
    _patch_runtime_with_service(monkeypatch, _MemoryCallingAgentService)
    tool = _delegated_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_k: None)

    result = await tool.run_json_async({"task": "run"})

    return result


async def test_child_runs_under_its_seat_when_the_service_knows_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_seats(monkeypatch, (frozenset({"1"}), None))
    session = _FakeStoreSession([_json_response({"results": []})])

    result = await _run_delegation_with_memory_call(monkeypatch, session)
    assert result == {"response": "child answer"}
    assert session.calls[-1]["headers"]["X-User-External-Id"] == "agent:1"  # type: ignore[index]


async def test_child_rejects_independent_mode_without_a_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A missing independent seat cannot borrow the parent identity.
    _patch_seats(monkeypatch, (frozenset({"2"}), None))
    session = _FakeStoreSession([_json_response({"results": []})])

    result = await _run_delegation_with_memory_call(monkeypatch, session)
    assert result == {"response": "child answer"}
    child = _RecordingAgentService.instances[-1]
    assert child.kwargs["memory"] is None
    assert child.kwargs["memory_available"] is False
    assert child.kwargs["memory_availability_reason"]
    assert session.calls == []


async def test_child_rejects_independent_mode_when_mapping_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.memory.remote import AGENT_SEATS_PROBE_FAILED_REASON

    _patch_seats(monkeypatch, (None, AGENT_SEATS_PROBE_FAILED_REASON))
    session = _FakeStoreSession([_json_response({"results": []})])

    result = await _run_delegation_with_memory_call(monkeypatch, session)
    assert result == {"response": "child answer"}
    child = _RecordingAgentService.instances[-1]
    assert child.kwargs["memory"] is None
    assert child.kwargs["memory_available"] is False
    assert child.kwargs["memory_availability_reason"]
    assert session.calls == []


async def test_a_raising_seat_lookup_never_retries_as_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(**_kwargs: object) -> object:
        raise RuntimeError("seat probe sentinel")

    _patch_seats(monkeypatch, explode)
    session = _FakeStoreSession([_json_response({"results": []})])

    result = await _run_delegation_with_memory_call(monkeypatch, session)
    assert result == {"response": "child answer"}
    child = _RecordingAgentService.instances[-1]
    assert child.kwargs["memory"] is None
    assert child.kwargs["memory_available"] is False
    assert child.kwargs["memory_availability_reason"]
    assert session.calls == []


async def test_a_non_remote_parent_store_never_triggers_the_seat_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.memory import remote as remote_module

    def forbidden(**_kwargs: object) -> object:
        pytest.fail("only the remote adapter can carry a seat identity")

    monkeypatch.setattr(remote_module, "registered_agent_seats", forbidden)
    _select_remote(monkeypatch)
    _patch_readiness(monkeypatch, (True, None))
    _patch_published_store(monkeypatch, object())

    result = await _run_delegation(monkeypatch)

    assert result == {"response": "child answer"}
    (child,) = _RecordingAgentService.instances
    assert child.kwargs["memory"] is not None
    assert child.kwargs["memory_enabled"] is True


@pytest.mark.parametrize("identity", [None, 0, "", " "])
def test_missing_delegating_identity_never_probes_readiness(
    monkeypatch: pytest.MonkeyPatch, identity: object
) -> None:
    from xagent.core.memory.remote import IDENTITY_REQUIRED_REASON

    _select_remote(monkeypatch)
    _patch_readiness(monkeypatch, (True, None))
    from xagent.core.memory import remote as remote_module

    monkeypatch.setattr(
        remote_module, "remote_memory_ready", lambda: pytest.fail("unexpected readyz")
    )
    assert _resolve_delegated_parent_memory(user_id=identity) == (
        None,
        False,
        IDENTITY_REQUIRED_REASON,
    )


def test_missing_independent_mapping_never_probes_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.memory import remote as remote_module
    from xagent.core.memory.remote import AGENT_SEAT_REQUIRED_REASON

    _select_remote(monkeypatch)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "1")
    from xagent.core.memory import agent_seats

    monkeypatch.setattr(agent_seats, "_provider", None)
    monkeypatch.setattr(
        remote_module, "remote_memory_ready", lambda: pytest.fail("unexpected readyz")
    )
    assert _resolve_delegated_parent_memory(user_id=1, agent_id=7) == (
        None,
        False,
        AGENT_SEAT_REQUIRED_REASON,
    )


@pytest.mark.asyncio
async def test_owned_binding_runs_independently_without_installed_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.memory import agent_seats
    from xagent.core.memory.remote import RemoteMemoryStore

    _select_remote(monkeypatch)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "true")
    monkeypatch.setattr(agent_seats, "_provider", None)
    monkeypatch.setenv(
        "XAGENT_MEMORY_AGENT_BINDINGS",
        json.dumps(
            [
                {
                    "user_id": 1,
                    "agent_id": 1,
                    "memory_user_id": 101,
                    "memory_agent_id": 202,
                    "memory_agent_owner_id": 101,
                    "service_url": "http://memsrv:8080",
                    "runtime_profile": hashlib.sha256(b"svc-key").hexdigest(),
                }
            ]
        ),
    )
    calls: list[tuple[str, str]] = []

    class Session(requests.Session):
        def request(self, *args: Any, **kwargs: Any) -> requests.Response:
            _method, url = args
            actor = kwargs["headers"]["X-User-External-Id"]
            calls.append((url, actor))
            response = requests.Response()
            response.status_code = 200
            payload = (
                {
                    "identity": {"id": 101 if actor == "1" else 202},
                    "scopes": ["memory:read", "memory:write"],
                }
                if url.endswith("/whoami")
                else {"results": []}
            )
            response._content = json.dumps(payload).encode()
            return response

    store = RemoteMemoryStore("http://memsrv:8080", "svc-key", session=Session())
    _patch_published_store(monkeypatch, store)
    _patch_readiness(monkeypatch, (True, None))

    async def execute(self: _RecordingAgentService, **kwargs: object) -> dict[str, str]:
        cast(RemoteMemoryStore, self.kwargs["memory"]).search(
            "synthetic delegated memory"
        )
        return {"output": "child answer"}

    monkeypatch.setattr(_RecordingAgentService, "execute_task", execute)
    await _run_delegation(monkeypatch)
    assert [actor for _, actor in calls] == ["1", "agent:1", "agent:1"]
    assert not any("/org/" in url for url, _ in calls)
    assert _RecordingAgentService.instances[0].kwargs["memory_enabled"] is True


@pytest.mark.asyncio
async def test_executable_foreign_owned_agent_gets_no_memory_or_parent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_remote(monkeypatch)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "true")
    _patch_published_store(
        monkeypatch, _forbidden_store("foreign memory must not be acquired")
    )
    _patch_readiness(monkeypatch, (True, None))
    await _run_delegation(monkeypatch, agent_owner_id=2)
    child = _RecordingAgentService.instances[0]
    assert child.kwargs["memory"] is None
    assert child.kwargs["memory_enabled"] is False
    assert (
        child.kwargs["memory_availability_reason"] == "agent_memory_ownership_required"
    )
