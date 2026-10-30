"""Real connector logic + synthetic SQLite/HTTP contracts; no live credentials."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests
from sqlalchemy import create_engine

from xagent.core.memory.publisher_service import build_publisher
from xagent.core.memory.seat_file_provider import FileAgentSeatProvider
from xagent.core.memory.seat_verifier import (
    MemsrvSeatVerifier,
    UserBinding,
    VerificationUnavailable,
)


@pytest.fixture
def verifier(tmp_path: Path) -> Any:
    engine = create_engine(f"sqlite:///{tmp_path / 'xagent.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE users(id INTEGER PRIMARY KEY,is_admin BOOLEAN)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE agents(id INTEGER PRIMARY KEY,user_id INTEGER,status TEXT,team_id INTEGER,visibility TEXT)"
        )
        connection.exec_driver_sql("INSERT INTO users VALUES (31,0)")
        connection.exec_driver_sql(
            "INSERT INTO agents VALUES (7,31,'published',NULL,'team')"
        )
    control, runtime = tmp_path / "control", tmp_path / "runtime"
    control.write_text("synthetic-control")
    runtime.write_text("synthetic-runtime")
    control.chmod(0o600)
    runtime.chmod(0o600)
    rows = [
        {
            "id": 200,
            "external_id": "31",
            "kind": "person",
            "status": "active",
            "role": "member",
            "department_id": 10,
            "accessible_users": [],
        },
        {
            "id": 201,
            "agent_ref": "7",
            "agent_owner_id": 200,
            "manager_id": 200,
            "kind": "agent",
            "status": "active",
            "role": "member",
            "department_id": 10,
            "accessible_users": [],
        },
        {
            "id": 999,
            "external_id": "unrelated",
            "kind": "person",
            "status": "active",
            "role": "member",
        },
    ]
    state: dict[str, Any] = {
        "rows": rows,
        "deny_agent": False,
        "org_status": 200,
        "admin": False,
        "scopes": ["memory:read", "memory:write"],
        "drift": False,
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def get(url: str, **kwargs: Any) -> requests.Response:
        calls.append((url, kwargs))
        assert kwargs["allow_redirects"] is False and kwargs["stream"] is True
        headers = kwargs["headers"]
        if url.endswith("/v1/org/users"):
            assert headers == {"Authorization": "Bearer synthetic-control"}
            payload, status = {"users": deepcopy(state["rows"])}, state["org_status"]
            if (
                state["drift"]
                and sum(call[0].endswith("/v1/org/users") for call in calls) == 2
            ):
                payload["users"][1]["agent_owner_id"] = 999
        else:
            assert url.endswith("/v1/whoami")
            assert headers["Authorization"] == "Bearer synthetic-runtime"
            identity = headers["X-User-External-Id"]
            assert identity in {"31", "agent:7"}
            payload = {
                "identity": {"id": 200 if identity == "31" else 201, "role": "member"},
                "kind": "user",
                "is_admin": state["admin"],
                "scopes": state["scopes"],
            }
            status = 401 if identity == "agent:7" and state["deny_agent"] else 200
        response = requests.Response()
        response.status_code = status
        response.raw = io.BytesIO(json.dumps(payload).encode())
        return response

    session = Mock(spec=requests.Session)
    session.get.side_effect = get
    source = MemsrvSeatVerifier(
        engine=engine,
        service_url="http://memory-internal:8080",
        control_key_file=control,
        runtime_key_file=runtime,
        bindings=(UserBinding(31, 200),),
        session=session,
    )
    try:
        yield source, state, calls, engine, control, runtime
    finally:
        engine.dispose()


def test_real_api_shapes_produce_only_verified_pinned_grants(verifier: Any) -> None:
    source, _state, calls, _engine, _control, _runtime = verifier
    grants = source()
    assert len(grants) == 1 and grants[0].agent_refs == frozenset({"7"})
    assert grants[0].request.service_url == "http://memory-internal:8080"
    assert grants[0].request.user_external_id == "31"
    assert len(calls) == 4
    assert not any("/memory/" in url or kwargs.get("json") for url, kwargs in calls)


@pytest.mark.parametrize(
    "change",
    [
        "person_mapping",
        "person_missing",
        "seat_missing",
        "owner",
        "disabled",
        "admin_seat",
        "selection",
        "org_scope",
        "runtime_admin",
        "runtime_scopes",
        "drift",
    ],
)
def test_no_grant_on_mapping_permission_or_selection_failure(
    verifier: Any, change: str
) -> None:
    source, state, _calls, _engine, _control, _runtime = verifier
    if change == "person_mapping":
        state["rows"][0]["external_id"] = "200"
    elif change == "person_missing":
        state["rows"] = state["rows"][1:]
    elif change == "seat_missing":
        state["rows"][1]["agent_ref"] = "different"
    elif change == "owner":
        state["rows"][1]["agent_owner_id"] = 999
    elif change == "disabled":
        state["rows"][1]["status"] = "disabled"
    elif change == "admin_seat":
        state["rows"][1]["role"] = "admin"
    elif change == "selection":
        state["deny_agent"] = True
    elif change == "org_scope":
        state["org_status"] = 403
    elif change == "runtime_admin":
        state["admin"] = True
    elif change == "runtime_scopes":
        state["scopes"].append("org:admin")
    else:
        state["drift"] = True
    with pytest.raises(VerificationUnavailable):
        source()


def test_policy_changes_change_fingerprint_but_unrelated_users_do_not(
    verifier: Any,
) -> None:
    source, state, _calls, _engine, _control, _runtime = verifier
    initial = source()[0]
    state["rows"][2]["status"] = "disabled"
    assert source()[0].authorization_fingerprint == initial.authorization_fingerprint
    state["rows"][1]["accessible_users"] = [{"user_id": 999, "mode": "read"}]
    assert source()[0].authorization_fingerprint != initial.authorization_fingerprint


def test_database_ownership_revocation_is_rechecked(verifier: Any) -> None:
    source, _state, _calls, engine, _control, _runtime = verifier
    assert source()[0].agent_refs == frozenset({"7"})
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA query_only=OFF")
        connection.exec_driver_sql("UPDATE agents SET status='archived' WHERE id=7")
        connection.commit()
    assert source()[0].agent_refs == frozenset()


def test_system_entry_builds_automatic_verifier_publisher(
    verifier: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _state, _calls, engine, control, runtime = verifier
    db = tmp_path / "database-url"
    db.write_text(str(engine.url))
    db.chmod(0o600)
    monkeypatch.setattr(requests, "Session", lambda: source.session)
    config = {
        "version": 1,
        "service_url": "http://memory-internal:8080",
        "control_key_file": str(control),
        "runtime_key_file": str(runtime),
        "database_url_file": str(db),
        "manifest": str(tmp_path / "grants.json"),
        "journal": str(tmp_path / "journal.json"),
        "bindings": [{"xagent_user_id": 31, "memory_user_id": 200}],
    }
    publisher, runtime_engine = build_publisher(config)
    publisher.start()
    reader = FileAgentSeatProvider(tmp_path / "grants.json")
    try:
        assert reader.refresh()
        snapshot = reader(source()[0].request)
        assert snapshot is not None and snapshot.agent_refs == frozenset({"7"})
    finally:
        reader.close()
        publisher.close()
        runtime_engine.dispose()
