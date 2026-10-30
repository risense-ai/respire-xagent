"""Trusted initialization records plus actual local SQL ownership, no publisher.

HTTP responses are synthetic contracts. Memory owner relations in the config
are explicitly fixture attestations, not inferred from whoami or execution ACL.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.memory import agent_seats
from xagent.core.memory.agent_bindings import approved_bindings
from xagent.core.memory.remote import (
    RemoteMemoryStore,
    RemoteMemoryStoreAuthenticationError,
    RemoteMemoryStoreError,
    RemoteMemoryStoreIdentityError,
    delegated_agent_identity,
)
from xagent.core.tools.adapters.vibe.agent_tool import _agent_memory_ownership_check
from xagent.core.user_context import current_user_id
from xagent.web.models.agent import Agent
from xagent.web.services.agent_team_scope import AgentTeamScope, owns_agent

URL, KEY = "http://memory-internal:8080", "fixture-runtime-only"
PROFILE = hashlib.sha256(KEY.encode()).hexdigest()


@pytest.fixture
def owned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    engine = create_engine(f"sqlite:///{tmp_path / 'xagent.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE users(id INTEGER PRIMARY KEY)")
        connection.exec_driver_sql(
            "CREATE TABLE agents(id INTEGER PRIMARY KEY,user_id INTEGER,status TEXT,team_id INTEGER,visibility TEXT)"
        )
        connection.exec_driver_sql("INSERT INTO users VALUES (31),(32)")
        connection.exec_driver_sql(
            "INSERT INTO agents VALUES (7,31,'published',10,'team')"
        )
    factory = sessionmaker(bind=engine)
    record = {
        "user_id": 31,
        "agent_id": 7,
        "memory_user_id": 101,
        "memory_agent_id": 202,
        "memory_agent_owner_id": 101,
        "service_url": URL,
        "runtime_profile": PROFILE,
    }
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", URL)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", KEY)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "true")
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", json.dumps([record]))
    monkeypatch.setattr(agent_seats, "_provider", None)
    calls: list[dict[str, Any]] = []
    state = {"person": 101, "agent": 202, "status": 200}

    def request(method: str, url: str, **kwargs: Any) -> requests.Response:
        calls.append({"method": method, "url": url, **kwargs})
        response = requests.Response()
        response.status_code = 200
        payload: dict[str, Any]
        if url.endswith("/v1/whoami"):
            response.status_code = state["status"]
            identity = kwargs["headers"].get("X-User-External-Id", "31")
            payload = {
                "scopes": ["memory:read", "memory:write"],
                "identity": {
                    "id": state["person"] if identity == "31" else state["agent"]
                },
            }
        else:
            response.status_code = state.get("data_status", 200)
            payload = {"results": [], "truncated": False}
            if response.status_code == 401:
                payload = {"error": {"code": "unauthorized", "message": "unauthorized"}}
                if state.get("malformed_error"):
                    payload = {"error": ["unexpected shape"]}
        response._content = json.dumps(payload).encode()
        return response

    session = Mock(spec=requests.Session)
    session.request.side_effect = request
    store = RemoteMemoryStore(URL, KEY, session=session)
    check = _agent_memory_ownership_check(31, 7, factory)
    token = current_user_id.set(31)
    try:
        yield engine, record, state, calls, store, check
    finally:
        current_user_id.reset(token)
        engine.dispose()


def test_owned_configured_agent_keeps_independent_identity_without_provider(
    owned: Any,
) -> None:
    _engine, _record, _state, calls, store, check = owned
    assert agent_seats._provider is None
    with delegated_agent_identity(7, store=store, owner_check=check):
        store.search("synthetic request")
    assert [call["headers"]["X-User-External-Id"] for call in calls] == [
        "31",
        "agent:7",
        "agent:7",
    ]
    assert all(call["headers"]["Authorization"] == f"Bearer {KEY}" for call in calls)
    assert not any("/org/" in call["url"] for call in calls)
    store.search("ordinary user after scope")
    assert calls[-1]["headers"]["X-User-External-Id"] == "31"


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("malformed_error", [True, False])
def test_authentication_failure_permanently_fences_independent_scope(
    owned: Any,
    monkeypatch: pytest.MonkeyPatch,
    configured: bool,
    malformed_error: bool,
) -> None:
    engine, _record, state, calls, store, check = owned
    if not configured:
        monkeypatch.delenv("XAGENT_MEMORY_AGENT_BINDINGS")
        registry = agent_seats.TrustedAgentSeatRegistry()
        registry.publish(
            service_url=URL,
            user_id=31,
            runtime_profile=PROFILE,
            agent_ids=[7],
            ttl_seconds=60,
            status="active",
        )
        monkeypatch.setattr(agent_seats, "_provider", registry)
    with delegated_agent_identity(7, store=store, owner_check=check):
        store.search("before remote identity rejection")
        answered = len(calls)
        state.update(data_status=401, malformed_error=malformed_error)
        with pytest.raises(RemoteMemoryStoreAuthenticationError) as rejected:
            store.search("identity disabled remotely")
        assert rejected.value.status_code == 401
        state["data_status"] = 200
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("restored identity cannot revive old scope")
        assert len(calls) == answered + 1  # no bare-key diagnostic or retry

        fresh = _agent_memory_ownership_check(31, 7, sessionmaker(bind=engine))
        with delegated_agent_identity(7, store=store, owner_check=fresh):
            store.search("explicit new scope after recovery")
        recovered = len(calls)
        assert recovered > answered + 1
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("fresh nested scope does not revive old scope")
        assert len(calls) == recovered


def test_business_permission_denial_does_not_invalidate_identity_scope(
    owned: Any,
) -> None:
    _engine, _record, state, calls, store, check = owned
    with delegated_agent_identity(7, store=store, owner_check=check):
        state["data_status"] = 403
        with pytest.raises(RemoteMemoryStoreError) as denied:
            store.search("resource forbidden")
        assert denied.value.status_code == 403
        assert not isinstance(denied.value, RemoteMemoryStoreAuthenticationError)
        answered = len(calls)
        state["data_status"] = 200
        store.search("another permitted operation in the same identity scope")
        assert len(calls) == answered + 1


def test_team_execution_permission_is_not_personal_memory_ownership(owned: Any) -> None:
    engine, _record, _state, calls, store, check = owned
    with engine.begin() as connection:
        connection.exec_driver_sql("UPDATE agents SET user_id=32 WHERE id=7")
    executable = Agent(id=7, user_id=32, team_id=10, visibility="team")
    assert owns_agent(executable, 31, AgentTeamScope(10, False))
    with (
        pytest.raises(RemoteMemoryStoreIdentityError),
        delegated_agent_identity(7, store=store, owner_check=check),
    ):
        pytest.fail("foreign-owned agent memory must not open")
    assert calls == []


@pytest.mark.parametrize(
    "change", ["foreign_owner", "deleted_agent", "deleted_user", "archived"]
)
def test_actual_db_revocation_and_restore_cannot_revive_old_scope(
    owned: Any, change: str
) -> None:
    engine, _record, _state, calls, store, check = owned
    with delegated_agent_identity(7, store=store, owner_check=check):
        store.search("before revocation")
        answered = len(calls)
        sql = {
            "foreign_owner": "UPDATE agents SET user_id=32 WHERE id=7",
            "deleted_agent": "DELETE FROM agents WHERE id=7",
            "deleted_user": "DELETE FROM users WHERE id=31",
            "archived": "UPDATE agents SET status='archived' WHERE id=7",
        }[change]
        with engine.begin() as connection:
            connection.exec_driver_sql(sql)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("revoked")
        assert len(calls) == answered
        with engine.begin() as connection:
            connection.exec_driver_sql("INSERT OR REPLACE INTO users VALUES (31)")
            connection.exec_driver_sql(
                "INSERT OR REPLACE INTO agents VALUES (7,31,'published',10,'team')"
            )
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("restored ownership cannot restore the old scope")
        assert len(calls) == answered

        fresh = _agent_memory_ownership_check(31, 7, sessionmaker(bind=engine))
        with delegated_agent_identity(7, store=store, owner_check=fresh):
            store.search("a fresh scope can use restored ownership")
        recovered = len(calls)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("old scope remains invalid after fresh nested scope")
        assert len(calls) == recovered


@pytest.mark.parametrize(
    "change",
    ["user", "missing_user", "mode", "url", "key", "mapping", "missing_mapping"],
)
def test_fixed_scope_rejects_changes_and_restoration(
    owned: Any, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _engine, record, _state, calls, store, check = owned
    with delegated_agent_identity(7, store=store, owner_check=check):
        store.search("before context change")
        answered = len(calls)
        if change in {"user", "missing_user"}:
            current_user_id.set(32 if change == "user" else None)
        elif change == "mode":
            monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "false")
        elif change == "url":
            monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://other")
        elif change == "key":
            monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "other-key")
        elif change == "mapping":
            monkeypatch.setenv(
                "XAGENT_MEMORY_AGENT_BINDINGS",
                json.dumps([{**record, "memory_agent_id": 303}]),
            )
        else:
            monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", "[]")
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("changed context")
        current_user_id.set(31)
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "true")
        monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", URL)
        monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", KEY)
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", json.dumps([record]))
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("restored context")
        assert len(calls) == answered


@pytest.mark.parametrize(
    "case",
    ["missing", "malformed", "owner_mismatch", "wrong_profile", "no_backend_guard"],
)
def test_missing_or_invalid_binding_denies_before_http(
    owned: Any, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    _engine, record, _state, calls, store, check = owned
    if case == "missing":
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", "[]")
    elif case == "malformed":
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", "{")
    elif case in {"owner_mismatch", "wrong_profile"}:
        changed = {
            **record,
            (
                "memory_agent_owner_id"
                if case == "owner_mismatch"
                else "runtime_profile"
            ): (999 if case == "owner_mismatch" else "0" * 64),
        }
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", json.dumps([changed]))
    with (
        pytest.raises(RemoteMemoryStoreIdentityError),
        delegated_agent_identity(
            7, store=store, owner_check=None if case == "no_backend_guard" else check
        ),
    ):
        pytest.fail("mapping must deny")
    assert calls == []


@pytest.mark.parametrize("case", ["person", "agent", "denied"])
def test_existing_whoami_must_select_the_pinned_initialization_ids(
    owned: Any, case: str
) -> None:
    _engine, _record, state, calls, store, check = owned
    state["status" if case == "denied" else case] = 401 if case == "denied" else 999
    with (
        pytest.raises(RemoteMemoryStoreIdentityError),
        delegated_agent_identity(7, store=store, owner_check=check),
    ):
        pytest.fail("wrong selector must not open memory")
    assert calls and all(call["url"].endswith("/v1/whoami") for call in calls)


def test_binding_parser_rejects_ambiguous_entries(
    owned: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _engine, record, _state, _calls, _store, _check = owned
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", json.dumps([record, record]))
    with pytest.raises(ValueError, match="ambiguous"):
        approved_bindings()


@pytest.mark.parametrize("change", ["seat_alias", "ref_alias"])
def test_binding_config_is_one_to_one(
    owned: Any, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _engine, record, _state, calls, store, check = owned
    alias = {
        **record,
        ("agent_id" if change == "seat_alias" else "user_id"): (
            8 if change == "seat_alias" else 32
        ),
    }
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", json.dumps([record, alias]))
    with (
        pytest.raises(RemoteMemoryStoreIdentityError),
        delegated_agent_identity(7, store=store, owner_check=check),
    ):
        pytest.fail("ambiguous binding must deny")
    assert calls == []


def test_unparseable_config_breaks_scope_permanently(
    owned: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _engine, record, _state, calls, store, check = owned
    with delegated_agent_identity(7, store=store, owner_check=check):
        answered = len(calls)
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", "[" * 2000 + "]" * 2000)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("config cannot be verified")
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", json.dumps([record]))
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("restoring config cannot revive this scope")
        assert len(calls) == answered


def test_unverifiable_backend_guard_cannot_resume_after_recovery(owned: Any) -> None:
    _engine, _record, _state, calls, store, check = owned
    healthy = [True]

    def guarded() -> None:
        check()
        if not healthy[0]:
            raise RuntimeError("synthetic backend unavailable")

    with delegated_agent_identity(7, store=store, owner_check=guarded):
        answered = len(calls)
        healthy[0] = False
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("backend failure")
        healthy[0] = True
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("old scope cannot resume")
        assert len(calls) == answered


@pytest.mark.parametrize("report", ["status", "get_store_info"])
@pytest.mark.parametrize("change", ["owner", "mapping"])
def test_configured_scope_guards_user_health_before_dependency_probes(
    owned: Any, monkeypatch: pytest.MonkeyPatch, report: str, change: str
) -> None:
    from xagent.core.memory.remote import RemoteMemoryReadiness
    from xagent.web import dynamic_memory_store as manager_module
    from xagent.web.revocable_memory_store import unwrap_memory_store

    engine, record, _state, calls, store, check = owned
    probes = []

    def readiness() -> RemoteMemoryReadiness:
        probes.append("readyz")
        return RemoteMemoryReadiness("ready", semantic_available=True)

    def upstream() -> tuple[str, str]:
        probes.append("upstream")
        return "ready", "synthetic upstream report"

    monkeypatch.setattr(manager_module, "remote_memory_readiness", readiness)
    monkeypatch.setattr(manager_module, "remote_upstream_report", upstream)
    manager = manager_module.DynamicMemoryStoreManager()
    manager.admit()
    published = unwrap_memory_store(manager.get_memory_store())
    assert isinstance(published, RemoteMemoryStore)
    published._session = store._session
    with delegated_agent_identity(7, store=published, owner_check=check):
        result = getattr(manager, report)()
        assert result.ready if report == "status" else result["state"] == "ready"
        answered, probed = len(calls), list(probes)
        if change == "owner":
            with engine.begin() as connection:
                connection.exec_driver_sql("UPDATE agents SET user_id=32 WHERE id=7")
        else:
            monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", "[]")
        with pytest.raises(RemoteMemoryStoreIdentityError):
            getattr(manager, report)()
        with engine.begin() as connection:
            connection.exec_driver_sql("UPDATE agents SET user_id=31 WHERE id=7")
        monkeypatch.setenv("XAGENT_MEMORY_AGENT_BINDINGS", json.dumps([record]))
        with pytest.raises(RemoteMemoryStoreIdentityError):
            getattr(manager, report)()
        assert len(calls) == answered
        assert probes == probed
