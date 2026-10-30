"""Authenticated Web/API to adapter contract with a synthetic upstream."""

from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.web import test_memory_api as fixtures
from xagent.web.api.memory import MemoryManagementRouter
from xagent.web.models.database import get_db
from xagent.web.models.user import User

# Reuse synthetic database/JWT fixtures without replacing authentication.
auth_headers = fixtures.auth_headers
regular_auth_headers = fixtures.regular_auth_headers
regular_user = fixtures.regular_user
test_db = fixtures.test_db


def test_authenticated_routes_drive_real_adapter_with_synthetic_contract(
    auth_headers: dict[str, str],
    regular_auth_headers: dict[str, str],
    test_db: User,
    regular_user: User,
) -> None:
    """JWT + DB + Web + adapter; upstream authorization here is synthetic."""
    import json
    from urllib.parse import urlsplit

    import requests

    from xagent.core.memory.remote import RemoteMemoryStore

    notes: dict[str, dict[str, Any]] = {}
    calls: list[tuple[str, str, dict[str, str]]] = []

    def request(method: str, url: str, **kwargs: Any) -> requests.Response:
        note: dict[str, Any] | None
        identity = kwargs["headers"]["X-User-External-Id"]
        calls.append((method, urlsplit(url).path, dict(kwargs["headers"])))
        path = urlsplit(url).path
        body = kwargs.get("json", {})
        status, payload = 200, {}
        if method == "POST" and path == "/v1/memory/notes":
            note = {**body, "owner": identity}
            notes[note["id"]] = note
            payload = {"id": note["id"], "success": True}
        elif path in {"/v1/memory/list", "/v1/memory/search"}:
            visible = [n for n in notes.values() if n["owner"] == identity]
            payload = {"results": visible, "truncated": False, "limit": 10_000}
        else:
            note_id = path.rsplit("/", 1)[-1]
            note = notes.get(note_id)
            if note is None or note["owner"] != identity:
                status, payload = (
                    403,
                    {"error": {"code": "forbidden", "message": "denied"}},
                )
            elif method == "GET":
                payload = note
            elif method == "PATCH":
                assert "id" not in body and "parent_id" not in body
                note.update(body)
                payload = {"id": note_id, "success": True}
            elif method == "DELETE":
                del notes[note_id]
                payload = {"success": True}
            else:
                pytest.fail(f"unexpected synthetic upstream operation: {method} {path}")
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode()
        return response

    transport = Mock(spec=requests.Session)
    transport.request.side_effect = request
    store = RemoteMemoryStore(
        "http://synthetic-memsrv", "runtime-only", session=transport
    )
    app = FastAPI()
    app.include_router(MemoryManagementRouter(lambda: store).get_router())
    app.dependency_overrides[get_db] = fixtures.override_get_db
    with TestClient(app) as api:
        assert api.get("/api/memory/list").status_code == 401
        assert calls == []
        forged = {
            **auth_headers,
            "X-User-External-Id": "agent:999",
            "X-Memory-Owner-Id": "999",
        }
        created = api.post(
            "/api/memory",
            json={"content": "synthetic lifecycle fixture"},
            headers=forged,
        )
        assert created.status_code == 200
        note_id = created.json()["id"]
        assert (
            api.get(f"/api/memory/{note_id}", headers=forged).json()["content"]
            == "synthetic lifecycle fixture"
        )
        assert (
            api.get(
                "/api/memory/list", params={"search": "fixture"}, headers=forged
            ).json()["total_count"]
            == 1
        )
        assert (
            api.put(
                f"/api/memory/{note_id}",
                json={"content": "synthetic edited"},
                headers=forged,
            ).status_code
            == 200
        )
        assert (
            api.get(f"/api/memory/{note_id}", headers=forged).json()["content"]
            == "synthetic edited"
        )
        answered = len(calls)
        assert (
            api.get(f"/api/memory/{note_id}", headers=regular_auth_headers).status_code
            == 404
        )
        assert len(calls) == answered + 1  # no retry as parent/key owner after denial
        assert calls[-1][2]["X-User-External-Id"] == str(regular_user.id)
        assert api.delete(f"/api/memory/{note_id}", headers=forged).status_code == 200
    assert notes == {}
    assert all(call[2]["Authorization"] == "Bearer runtime-only" for call in calls)
    assert all("X-Memory-Owner-Id" not in call[2] for call in calls)
    assert all(call[2]["X-User-External-Id"] == str(test_db.id) for call in calls[:-2])
    assert not any(method == "PUT" for method, _, _ in calls)


def test_remote_list_and_stats_use_existing_visible_acl(
    auth_headers: dict[str, str],
    test_db: User,
) -> None:
    import json

    import requests

    from xagent.core.memory.remote import RemoteMemoryStore

    calls: list[dict[str, Any]] = []
    notes = [
        {"id": "own", "content": "synthetic own", "space_kind": "personal"},
        {
            "id": "dept",
            "content": "synthetic department",
            "space_kind": "department",
            "space_id": 10,
        },
        {"id": "company", "content": "synthetic company", "space_kind": "company"},
    ]

    def request(method: str, url: str, **kwargs: Any) -> requests.Response:
        calls.append(kwargs)
        if url.endswith("/list"):
            assert method == "POST" and kwargs["json"]["scope"] == "visible"
            payload = {"results": notes, "truncated": False, "scope": "visible"}
        else:
            assert method == "GET" and url.endswith("/stats")
            assert kwargs["params"] == {"scope": "visible"}
            payload = {
                "total_count": 3,
                "scope": "visible",
                "category_counts": {},
                "tag_counts": {},
            }
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        return response

    transport = Mock(spec=requests.Session)
    transport.request.side_effect = request
    adapter = RemoteMemoryStore(
        "http://memory-internal:8080", "runtime-only", session=transport
    )
    app = FastAPI()
    app.include_router(MemoryManagementRouter(lambda: adapter).get_router())
    app.dependency_overrides[get_db] = fixtures.override_get_db
    with TestClient(app) as api:
        listing = api.get("/api/memory/list", headers=auth_headers)
        stats = api.get("/api/memory/stats", headers=auth_headers)
        assert listing.status_code == stats.status_code == 200
        assert listing.json()["scope"] == stats.json()["scope"] == "visible"
        assert listing.json()["total_count"] == stats.json()["total_count"] == 3
    assert len(calls) == 2
    assert all(
        call["headers"]["X-User-External-Id"] == str(test_db.id) for call in calls
    )
    assert all(
        call["headers"]["Authorization"] == "Bearer runtime-only" for call in calls
    )
