"""Contract tests for the remote memory adapter (方案 §3.1/§3.5).

The HTTP layer is faked, so these tests never touch the network (pytest-socket
is active in this repository). They pin:

* the wire shape of all nine ``MemoryStore`` methods plus the two scope
  overrides: method, path, headers, body, and the response-to-``MemoryNote``
  reconstruction;
* the §3.5 semantic equivalences: raise-don't-swallow on service failure,
  byte-identical denial message, bytes/base64 round-trip, naive timestamps,
  clear/stats scoped to the caller, filter pass-through, no write retries;
* identity header behaviour with and without an active ``current_user_id``;
* the T14 §B1 additions: the ``[to:<target>]`` delivery marker (and every shape
  that must stay a normal write), the explicit-owner header, and the cached
  ``/readyz`` probe behind the task-startup memory policy.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from typing import Any, List, Optional, cast

import pytest
import requests

from xagent.core.execution_scope import (
    ExecutionScope,
    reset_execution_scope,
    set_execution_scope,
)
from xagent.core.memory import remote as remote_module
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.remote import (
    AGENT_SEATS_PROBE_FAILED_REASON,
    NOT_FOUND_OR_DENIED_ERROR,
    READINESS_UNAVAILABLE_REASON,
    READINESS_UNCONFIGURED_REASON,
    READY_PATH,
    RemoteMemoryStore,
    RemoteMemoryStoreAuthenticationError,
    RemoteMemoryStoreConfigurationError,
    RemoteMemoryStoreError,
    RemoteMemoryStoreIdentityError,
    delegated_agent_identity,
    explicit_memory_owner,
    registered_agent_seats,
    remote_memory_ready,
    remote_memory_store_from_config,
)
from xagent.core.memory.scope_columns import SCOPE_EXCLUSIVE_FILTER_KEY
from xagent.core.user_context import current_user_id

SERVICE_URL = "http://memsrv:8080"
SERVICE_KEY = "svc-key"
SERVICE_PROFILE = hashlib.sha256(SERVICE_KEY.encode()).hexdigest()


def _response(
    status: int = 200, payload: Any = None, *, raw: Optional[str] = None
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    body = (
        raw if raw is not None else json.dumps(payload if payload is not None else {})
    )
    response._content = body.encode("utf-8")  # noqa: SLF001 - fake transport
    response.headers["Content-Type"] = "application/json"
    return response


class FakeSession:
    """Records requests and replays scripted responses in order."""

    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.calls: List[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return cast(requests.Response, item)


class RecordingStore(RemoteMemoryStore):
    """The adapter with a scripted fake transport."""

    def __init__(self, *responses: Any, user_id: Optional[int] = 23) -> None:
        self.session = FakeSession(list(responses))
        super().__init__(
            SERVICE_URL,
            SERVICE_KEY,
            session=self.session,  # type: ignore[arg-type]
            user_id_provider=lambda: user_id,
        )


def _note(**overrides: Any) -> MemoryNote:
    base: dict[str, Any] = {
        "id": "note-1",
        "content": "hello",
        "keywords": ["k1"],
        "tags": ["t1"],
        "category": "general",
        "metadata": {"source": "test"},
    }
    base.update(overrides)
    return MemoryNote(**base)


@pytest.fixture
def scope() -> Any:
    """Activate an execution scope for the duration of one test."""
    token = set_execution_scope(
        ExecutionScope(
            memory_dimensions={"client_application_id": "42"},
            strict_memory_isolation=False,
        )
    )
    try:
        yield
    finally:
        reset_execution_scope(token)


@pytest.fixture
def strict_scope() -> Any:
    token = set_execution_scope(
        ExecutionScope(memory_dimensions={}, strict_memory_isolation=True)
    )
    try:
        yield
    finally:
        reset_execution_scope(token)


# ---------------------------------------------------------------------------
# add / get / update / delete
# ---------------------------------------------------------------------------


def test_add_posts_note_and_returns_service_id() -> None:
    store = RecordingStore(_response(200, {"id": "server-id"}))

    response = store.add(_note())

    assert response.success is True
    assert response.memory_id == "server-id"
    call = store.session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/notes"
    body = call["json"]
    assert body["id"] == "note-1"
    assert body["content"] == "hello"
    assert body["keywords"] == ["k1"]
    assert body["tags"] == ["t1"]
    assert body["category"] == "general"
    assert body["mime_type"] == "text/plain"
    assert body["metadata"] == {"source": "test"}


def test_add_drops_caller_supplied_user_id() -> None:
    """Ownership is server-side (§3.4.3): a stale stamp must not be forwarded."""
    store = RecordingStore(_response(200, {"id": "n"}))

    store.add(_note(metadata={"user_id": 7, "source": "test"}))

    assert store.session.calls[-1]["json"]["metadata"] == {"source": "test"}


def test_add_stamps_execution_scope_dimensions(scope: Any) -> None:
    store = RecordingStore(_response(200, {"id": "n"}))

    store.add(_note())

    metadata = store.session.calls[-1]["json"]["metadata"]
    assert metadata["execution_scope_client_application_id"] == "42"


def test_add_encodes_binary_content_as_base64() -> None:
    store = RecordingStore(_response(200, {}), _response(200, {}))

    store.add(_note(content=b"\x00\x01binary"))

    body = store.session.calls[-1]["json"]
    assert "content" not in body
    assert body["content_b64"] == base64.b64encode(b"\x00\x01binary").decode("ascii")
    # An empty body is tolerated: the id is client-side.
    assert store.add(_note()).success is True


def test_remember_posts_the_judged_write_and_returns_the_branch() -> None:
    """The judged write goes to /remember and reports merge / parent / create."""
    store = RecordingStore(
        _response(
            200,
            {"action": "parent", "id": "server-id", "parent_id": "parent-1"},
        )
    )

    response = store.remember(_note())

    assert response.success is True
    assert response.memory_id == "server-id"
    assert response.metadata == {"action": "parent", "parent_id": "parent-1"}
    call = store.session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/remember"
    body = call["json"]
    assert body["content"] == "hello"
    assert body["category"] == "general"
    assert body["keywords"] == ["k1"]
    assert body["metadata"] == {"source": "test"}
    assert body["force"] is False
    # The service assigns the id; sending a client uuid would only risk a 409.
    assert "id" not in body


def test_remember_forwards_force_kind_and_drops_user_id() -> None:
    store = RecordingStore(_response(200, {"action": "merge", "id": "n"}))

    response = store.remember(
        _note(metadata={"user_id": 7, "kind": "domain_insight"}), force=True
    )

    body = store.session.calls[-1]["json"]
    assert body["force"] is True
    assert body["kind"] == "domain_insight"
    assert body["metadata"] == {"kind": "domain_insight"}
    assert response.metadata == {"action": "merge", "parent_id": None}


def test_remember_stamps_execution_scope_dimensions(scope: Any) -> None:
    store = RecordingStore(_response(200, {"action": "create", "id": "n"}))

    store.remember(_note())

    metadata = store.session.calls[-1]["json"]["metadata"]
    assert metadata["execution_scope_client_application_id"] == "42"


def test_remember_raises_on_a_rejected_write() -> None:
    store = RecordingStore(_response(400, {"error": {"code": "bad_request"}}))

    with pytest.raises(RemoteMemoryStoreError):
        store.remember(_note())


def test_get_returns_memory_note_instance_with_naive_timestamp() -> None:
    store = RecordingStore(
        _response(
            200,
            {
                "id": "note-1",
                "content": "hello",
                "keywords": ["k1"],
                "tags": ["t1"],
                "category": "general",
                "timestamp": "2026-01-02T03:04:05",
                "mime_type": "text/plain",
                "metadata": {"user_id": 7},
            },
        )
    )

    response = store.get("note-1")

    assert response.success is True
    assert isinstance(response.content, MemoryNote)
    assert response.content.id == "note-1"
    assert response.content.content == "hello"
    assert response.content.metadata == {"user_id": 7}
    assert response.content.timestamp == datetime(2026, 1, 2, 3, 4, 5)
    assert response.content.timestamp.tzinfo is None
    call = store.session.calls[-1]
    assert call["method"] == "GET"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/notes/note-1"


def test_get_decodes_content_b64_and_normalizes_aware_timestamp() -> None:
    store = RecordingStore(
        _response(
            200,
            {
                "id": "note-1",
                "content_b64": base64.b64encode(b"\x00\x01").decode("ascii"),
                "timestamp": "2026-01-02T03:04:05+08:00",
            },
        )
    )

    note = store.get("note-1").content

    assert isinstance(note, MemoryNote)
    assert note.content == b"\x00\x01"
    assert note.timestamp == datetime(2026, 1, 2, 3, 4, 5)


@pytest.mark.parametrize("status", [403, 404])
def test_get_denied_returns_the_local_denial_message(status: int) -> None:
    store = RecordingStore(_response(status, {"error": {"code": "denied"}}))

    response = store.get("note-1")

    assert response.success is False
    assert response.error == NOT_FOUND_OR_DENIED_ERROR
    assert response.memory_id == "note-1"


def test_update_puts_the_note() -> None:
    store = RecordingStore(_response(200, {"id": "note-1"}))

    response = store.update(_note())

    assert response.success is True
    call = store.session.calls[-1]
    assert call["method"] == "PUT"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/notes/note-1"
    assert call["json"]["content"] == "hello"


def test_update_denied_returns_denial_message() -> None:
    store = RecordingStore(_response(404, {}))

    response = store.update(_note())

    assert response.success is False
    assert response.error == NOT_FOUND_OR_DENIED_ERROR


def test_update_checks_scope_before_writing(strict_scope: Any) -> None:
    """A note outside the active scope must not reach the service as a write."""
    store = RecordingStore(
        _response(
            200,
            {"id": "note-1", "content": "x", "metadata": {"execution_scope_a": "1"}},
        )
    )

    response = store.update(_note())

    assert response.success is False
    assert response.error == NOT_FOUND_OR_DENIED_ERROR
    assert [call["method"] for call in store.session.calls] == ["GET"]


def test_delete_deletes_by_id() -> None:
    store = RecordingStore(_response(204))

    response = store.delete("note-1")

    assert response.success is True
    assert response.memory_id == "note-1"
    call = store.session.calls[-1]
    assert call["method"] == "DELETE"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/notes/note-1"


def test_delete_missing_returns_denial_message() -> None:
    store = RecordingStore(_response(404, {}))

    response = store.delete("note-1")

    assert response.success is False
    assert response.error == NOT_FOUND_OR_DENIED_ERROR
    assert response.memory_id == "note-1"


def test_get_url_quotes_the_note_id() -> None:
    store = RecordingStore(_response(404, {}))

    store.get("a/b c")

    assert store.session.calls[-1]["url"] == f"{SERVICE_URL}/v1/memory/notes/a%2Fb%20c"


# ---------------------------------------------------------------------------
# search / list_all / stats / clear
# ---------------------------------------------------------------------------


def test_search_forwards_query_filters_and_threshold() -> None:
    filters = {"category": "general", "tags": ["t1"], "metadata": {"user_id": 3}}
    store = RecordingStore(
        _response(200, [{"id": "note-1", "content": "hello", "category": "general"}])
    )

    notes = store.search("hello", k=7, filters=filters, similarity_threshold=0.5)

    assert [note.id for note in notes] == ["note-1"]
    assert isinstance(notes[0], MemoryNote)
    call = store.session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/search"
    assert call["json"] == {
        "query": "hello",
        "k": 7,
        "filters": filters,
        "similarity_threshold": 0.5,
    }
    # The caller's dict is never mutated (no user_id filter is injected).
    assert filters == {
        "category": "general",
        "tags": ["t1"],
        "metadata": {"user_id": 3},
    }


def test_search_merges_scope_dimensions_into_metadata_filter(scope: Any) -> None:
    store = RecordingStore(_response(200, []))

    store.search("hello", filters={"category": "general"})

    assert store.session.calls[-1]["json"]["filters"] == {
        "category": "general",
        "metadata": {"execution_scope_client_application_id": "42"},
    }


def test_search_adds_scope_exclusive_directive(strict_scope: Any) -> None:
    store = RecordingStore(_response(200, []))

    store.search("hello")

    assert store.session.calls[-1]["json"]["filters"] == {
        SCOPE_EXCLUSIVE_FILTER_KEY: True
    }


def test_search_accepts_wrapped_result_payload() -> None:
    store = RecordingStore(_response(200, {"results": [{"id": "n1", "content": "x"}]}))

    assert [note.id for note in store.search("x")] == ["n1"]


def test_search_empty_array_is_a_real_empty_result() -> None:
    store = RecordingStore(_response(200, []))

    assert store.search("nothing") == []


def test_list_all_posts_filters() -> None:
    store = RecordingStore(_response(200, {"memories": [{"id": "n1", "content": "x"}]}))

    notes = store.list_all({"category": "general"})

    assert [note.id for note in notes] == ["n1"]
    call = store.session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/list"
    assert call["json"] == {"filters": {"category": "general"}}


def test_get_stats_normalizes_the_contract_keys() -> None:
    store = RecordingStore(
        _response(200, {"category_counts": {"general": 2, "decision": 1}})
    )

    stats = store.get_stats()

    assert stats["total_count"] == 3
    assert stats["category_counts"] == {"general": 2, "decision": 1}
    assert stats["tag_counts"] == {}
    assert stats["memory_store_type"] == "remote"
    call = store.session.calls[-1]
    assert call["method"] == "GET"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/stats"
    assert call["params"] is None


def test_get_stats_keeps_service_reported_store_type() -> None:
    store = RecordingStore(
        _response(200, {"total_count": 5, "memory_store_type": "memsrv"})
    )

    assert store.get_stats()["memory_store_type"] == "memsrv"


def test_clear_posts_without_body() -> None:
    store = RecordingStore(_response(200, {}))

    store.clear()

    call = store.session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/clear"
    assert call["json"] is None


# ---------------------------------------------------------------------------
# scope maintenance endpoints
# ---------------------------------------------------------------------------


def test_delete_by_scope_dimension_posts_predicate_and_returns_count() -> None:
    store = RecordingStore(_response(200, {"deleted_count": 3}))

    response = store.delete_by_scope_dimension("client_application_id", 42)

    assert response.success is True
    assert response.metadata == {"deleted_count": 3}
    call = store.session.calls[-1]
    assert call["url"] == f"{SERVICE_URL}/v1/memory/scope/delete-by-dimension"
    assert call["json"] == {"dim_key": "client_application_id", "value": "42"}


def test_delete_by_scope_dimension_403_raises() -> None:
    store = RecordingStore(_response(403, {"error": {"code": "forbidden"}}))

    with pytest.raises(RemoteMemoryStoreError) as error:
        store.delete_by_scope_dimension("client_application_id", "42")

    assert error.value.status_code == 403


def test_list_scope_dimension_values_returns_a_set() -> None:
    store = RecordingStore(_response(200, {"values": ["1", "2", "2"]}))

    assert store.list_scope_dimension_values("client_application_id") == {"1", "2"}
    call = store.session.calls[-1]
    assert call["method"] == "GET"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/scope/values"
    assert call["params"] == {"dim": "client_application_id"}


# ---------------------------------------------------------------------------
# failure semantics (§3.5.1/§3.5.2/§3.5.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 503])
def test_search_service_failure_raises(status: int) -> None:
    store = RecordingStore(_response(status, {"error": {"code": "boom"}}))

    with pytest.raises(RemoteMemoryStoreError):
        store.search("hello")


def test_list_all_service_failure_raises() -> None:
    store = RecordingStore(_response(500, {}))

    with pytest.raises(RemoteMemoryStoreError):
        store.list_all()


def test_get_5xx_raises_instead_of_reporting_a_miss() -> None:
    store = RecordingStore(_response(500, {}))

    with pytest.raises(RemoteMemoryStoreError):
        store.get("note-1")


def test_timeout_raises_adapter_error() -> None:
    store = RecordingStore(requests.exceptions.Timeout("slow"))

    with pytest.raises(RemoteMemoryStoreError):
        store.search("hello")


def test_connection_failure_raises_adapter_error() -> None:
    store = RecordingStore(requests.exceptions.ConnectionError("refused"))

    with pytest.raises(RemoteMemoryStoreError):
        store.list_all()


def test_401_raises_authentication_error() -> None:
    store = RecordingStore(_response(401, {"error": {"code": "unauthorized"}}))

    with pytest.raises(RemoteMemoryStoreAuthenticationError):
        store.search("hello")


def test_non_json_success_payload_raises() -> None:
    store = RecordingStore(_response(200, raw="<html>gateway</html>"))

    with pytest.raises(RemoteMemoryStoreError):
        store.search("hello")


def test_writes_are_not_retried() -> None:
    """§3.5.8: a failed write must surface, never be replayed."""
    store = RecordingStore(_response(500, {}))

    with pytest.raises(RemoteMemoryStoreError):
        store.add(_note())

    assert len(store.session.calls) == 1


def test_read_timeout_is_tighter_than_write_timeout() -> None:
    store = RecordingStore(_response(200, []), _response(200, {}))

    store.search("hello")
    store.add(_note())

    assert store.session.calls[0]["timeout"] == (2.0, 2.0)
    assert store.session.calls[1]["timeout"] == (2.0, 5.0)


# ---------------------------------------------------------------------------
# identity headers (§3.4)
# ---------------------------------------------------------------------------


def test_identity_header_is_sent_with_the_contextvar_user() -> None:
    session = FakeSession([_response(200, [])])
    token = current_user_id.set(12345)
    try:
        store = RemoteMemoryStore(
            SERVICE_URL,
            SERVICE_KEY,
            session=session,  # type: ignore[arg-type]
        )
        store.search("hello")
    finally:
        current_user_id.reset(token)

    headers = session.calls[-1]["headers"]
    assert headers["Authorization"] == f"Bearer {SERVICE_KEY}"
    assert headers["X-User-External-Id"] == "12345"


def test_missing_identity_is_rejected_without_http() -> None:
    session = FakeSession([_response(200, [])])
    token = current_user_id.set(None)
    try:
        store = RemoteMemoryStore(
            SERVICE_URL,
            SERVICE_KEY,
            session=session,  # type: ignore[arg-type]
        )
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("hello")
    finally:
        current_user_id.reset(token)

    assert session.calls == []


# ---------------------------------------------------------------------------
# configuration factory (§3.3 row 6)
# ---------------------------------------------------------------------------


def test_factory_is_inactive_without_the_backend_switch(monkeypatch: Any) -> None:
    monkeypatch.delenv("XAGENT_MEMORY_BACKEND", raising=False)
    monkeypatch.delenv("XAGENT_MEMORY_SERVICE_URL", raising=False)
    monkeypatch.delenv("XAGENT_MEMORY_SERVICE_KEY", raising=False)

    assert remote_memory_store_from_config() is None


def test_factory_builds_a_store_when_selected(monkeypatch: Any) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", SERVICE_URL)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", SERVICE_KEY)

    store = remote_memory_store_from_config()

    assert isinstance(store, RemoteMemoryStore)
    assert store._base_url == SERVICE_URL  # noqa: SLF001 - config wiring assertion


def test_factory_rejects_an_incomplete_configuration(monkeypatch: Any) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", SERVICE_URL)
    monkeypatch.delenv("XAGENT_MEMORY_SERVICE_KEY", raising=False)

    with pytest.raises(RemoteMemoryStoreConfigurationError):
        remote_memory_store_from_config()


# ---------------------------------------------------------------------------
# T14 §B1-1: the directed delivery marker
# ---------------------------------------------------------------------------


def test_add_delivers_instead_of_writing_when_the_marker_names_a_target() -> None:
    store = RecordingStore(_response(200, {"id": "delivered-1"}))

    response = store.add(_note(content="[to:userb]   hello from A"))

    assert response.success is True
    assert response.memory_id == "delivered-1"
    call = store.session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/deliver"
    body = call["json"]
    assert body["target"] == "userb"
    assert body["content"] == "hello from A"
    assert body["category"] == "general"
    assert body["metadata"] == {"source": "test"}
    assert body["keywords"] == ["k1"]
    assert body["tags"] == ["t1"]
    assert "id" not in body


def test_delivery_strips_the_marker_and_a_following_newline() -> None:
    store = RecordingStore(_response(200, {}))

    store.add(_note(content="[to:userb]\nhello"))

    assert store.session.calls[-1]["json"]["content"] == "hello"


def test_share_dept_marker_plants_note_in_own_department() -> None:
    store = RecordingStore(_response(200, {"id": "shared-1"}))

    store.add(_note(content="[share:dept] team lesson learned"))

    call = store.session.calls[-1]
    assert call["url"] == f"{SERVICE_URL}/v1/memory/notes"
    body = call["json"]
    assert body["content"] == "team lesson learned"
    assert body["space_kind"] == "department"
    assert "space_id" not in body  # service resolves the caller's own department


def test_share_marker_accepts_explicit_department_and_company() -> None:
    store = RecordingStore(_response(200, {}), _response(200, {}))

    store.add(_note(content="[share:dept:2] platform note"))
    assert store.session.calls[-1]["json"]["space_kind"] == "department"
    assert store.session.calls[-1]["json"]["space_id"] == 2

    store.add(_note(content="[share:company] all-hands notice"))
    body = store.session.calls[-1]["json"]
    assert body["space_kind"] == "company"
    assert "space_id" not in body
    assert body["content"] == "all-hands notice"


def test_share_marker_never_matches_without_a_body() -> None:
    store = RecordingStore(_response(200, {}))

    store.add(_note(content="[share:dept]"))
    body = store.session.calls[-1]["json"]
    assert "space_kind" not in body
    assert body["content"] == "[share:dept]"


def test_delivery_marker_takes_precedence_over_share() -> None:
    store = RecordingStore(_response(200, {}))

    store.add(_note(content="[to:userb] [share:dept] not shared"))
    body = store.session.calls[-1]["json"]
    assert store.session.calls[-1]["url"].endswith("/v1/memory/deliver")
    assert body["content"] == "[share:dept] not shared"


def test_delivery_never_forwards_a_caller_supplied_user_id() -> None:
    store = RecordingStore(_response(200, {}))

    store.add(_note(content="[to:userb] hi", metadata={"user_id": 7}))

    assert store.session.calls[-1]["json"]["metadata"] == {}


def test_delivery_carries_the_active_scope_dimensions(scope: Any) -> None:
    """The channel is transparent: the caller's stamps ride along untouched."""
    store = RecordingStore(_response(200, {}))

    store.add(_note(content="[to:userb] hi"))

    metadata = store.session.calls[-1]["json"]["metadata"]
    assert metadata["execution_scope_client_application_id"] == "42"


def test_delivery_omits_empty_keyword_and_tag_lists() -> None:
    store = RecordingStore(_response(200, {}))

    store.add(_note(content="[to:userb] hi", keywords=[], tags=[]))

    body = store.session.calls[-1]["json"]
    assert "keywords" not in body
    assert "tags" not in body


def test_delivery_target_is_trimmed() -> None:
    store = RecordingStore(_response(200, {}))

    store.add(_note(content="[to: userb ] hi"))

    assert store.session.calls[-1]["json"]["target"] == "userb"


@pytest.mark.parametrize(
    "content",
    [
        "plain note",
        "[to:] hi",
        "[to: ] hi",
        "see [to:userb] for context",
        "[TO:userb] hi",
    ],
    ids=("plain", "empty-target", "blank-target", "inline-marker", "wrong-case"),
)
def test_unmarked_content_keeps_the_notes_path(content: str) -> None:
    store = RecordingStore(_response(200, {"id": "n"}))

    store.add(_note(content=content))

    assert store.session.calls[-1]["url"] == f"{SERVICE_URL}/v1/memory/notes"
    assert store.session.calls[-1]["json"]["content"] == content


def test_binary_content_is_never_read_as_a_marker() -> None:
    store = RecordingStore(_response(200, {}))

    store.add(_note(content=b"[to:userb] hi"))

    body = store.session.calls[-1]["json"]
    assert store.session.calls[-1]["url"] == f"{SERVICE_URL}/v1/memory/notes"
    assert "content" not in body


def test_delivery_failure_raises_instead_of_reporting_a_written_note() -> None:
    """A service that has no deliver endpoint yet must not fake a write."""
    store = RecordingStore(_response(405, {"error": {"code": "method_not_allowed"}}))

    with pytest.raises(RemoteMemoryStoreError) as error:
        store.add(_note(content="[to:userb] hi"))

    assert error.value.status_code == 405


def test_delivery_to_an_unknown_target_raises_with_the_service_status() -> None:
    store = RecordingStore(_response(404, {"error": {"code": "not_found"}}))

    with pytest.raises(RemoteMemoryStoreError) as error:
        store.add(_note(content="[to:nobody] hi"))

    assert error.value.status_code == 404


def test_delivery_authentication_failure_raises_authentication_error() -> None:
    store = RecordingStore(_response(401, {"error": {"code": "unauthorized"}}))

    with pytest.raises(RemoteMemoryStoreAuthenticationError):
        store.add(_note(content="[to:userb] hi"))


# ---------------------------------------------------------------------------
# T14 §B1-2: the explicit owner header
# ---------------------------------------------------------------------------


def test_update_sends_the_explicit_owner_header() -> None:
    store = RecordingStore(_response(200, {"id": "note-1"}))

    store.update(_note(), owner_id=42)

    headers = store.session.calls[-1]["headers"]
    assert headers["X-Memory-Owner-Id"] == "42"


def test_update_without_an_explicit_owner_sends_no_owner_header() -> None:
    store = RecordingStore(_response(200, {"id": "note-1"}))

    store.update(_note())

    assert "X-Memory-Owner-Id" not in store.session.calls[-1]["headers"]


def test_delete_sends_the_owner_header_and_keeps_the_identity_header() -> None:
    session = FakeSession([_response(204)])
    token = current_user_id.set(9)
    try:
        store = RemoteMemoryStore(
            SERVICE_URL,
            SERVICE_KEY,
            session=session,  # type: ignore[arg-type]
        )
        store.delete("note-1", owner_id=42)
    finally:
        current_user_id.reset(token)

    headers = session.calls[-1]["headers"]
    assert headers["X-Memory-Owner-Id"] == "42"
    assert headers["X-User-External-Id"] == "9"


def test_owner_context_manager_scopes_the_header_to_the_block() -> None:
    store = RecordingStore(_response(204), _response(204))

    with explicit_memory_owner(7):
        store.delete("note-1")
    store.delete("note-2")

    assert store.session.calls[0]["headers"]["X-Memory-Owner-Id"] == "7"
    assert "X-Memory-Owner-Id" not in store.session.calls[1]["headers"]


def test_owner_context_manager_with_none_is_a_no_op() -> None:
    store = RecordingStore(_response(204))

    with explicit_memory_owner(None):
        store.delete("note-1")

    assert "X-Memory-Owner-Id" not in store.session.calls[-1]["headers"]


def test_explicit_owner_parameter_wins_over_the_ambient_context() -> None:
    store = RecordingStore(_response(204))

    with explicit_memory_owner(7):
        store.delete("note-1", owner_id=42)

    assert store.session.calls[-1]["headers"]["X-Memory-Owner-Id"] == "42"


def test_explicit_owner_denial_keeps_the_adapter_denial_message() -> None:
    store = RecordingStore(_response(403, {"error": {"code": "forbidden"}}))

    response = store.delete("note-1", owner_id=42)

    assert response.success is False
    assert response.error == NOT_FOUND_OR_DENIED_ERROR


@pytest.mark.parametrize("status", [403, 404])
def test_explicit_owner_update_denial_keeps_the_adapter_denial_message(
    status: int,
) -> None:
    store = RecordingStore(_response(status, {}))

    response = store.update(_note(), owner_id=42)

    assert response.success is False
    assert response.error == NOT_FOUND_OR_DENIED_ERROR


# ---------------------------------------------------------------------------
# T14 §B1-3: the readiness probe behind the memory policy
# ---------------------------------------------------------------------------


class FakeProbeSession:
    """Replays one scripted readiness answer; ``get`` only."""

    def __init__(self, answer: Any) -> None:
        self._answer = answer
        self.calls: List[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self._answer, Exception):
            raise self._answer
        return cast(requests.Response, self._answer)


@pytest.fixture
def configured_remote_service(monkeypatch: Any) -> Any:
    """Configure the remote service and clear the process readiness cache."""
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", SERVICE_URL)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", SERVICE_KEY)
    monkeypatch.setattr(remote_module, "_readiness_cache", None)
    yield
    monkeypatch.setattr(remote_module, "_readiness_cache", None)


def test_readiness_probe_reports_a_ready_service(
    configured_remote_service: Any,
) -> None:
    session = FakeProbeSession(_response(200, {"status": "ok"}))

    available, reason = remote_memory_ready(session=session)

    assert available is True
    assert reason is None
    call = session.calls[-1]
    assert call["url"] == f"{SERVICE_URL}{READY_PATH}"
    assert call["headers"]["Authorization"] == f"Bearer {SERVICE_KEY}"
    assert call["timeout"] == (2.0, 2.0)


@pytest.mark.parametrize("status", [500, 503])
def test_readiness_probe_reports_an_unavailable_service(
    configured_remote_service: Any, status: int
) -> None:
    session = FakeProbeSession(_response(status, {"status": "unavailable"}))

    available, reason = remote_memory_ready(session=session)

    assert available is False
    assert reason == READINESS_UNAVAILABLE_REASON


def test_readiness_probe_reports_a_transport_failure(
    configured_remote_service: Any,
) -> None:
    session = FakeProbeSession(requests.exceptions.ConnectionError("refused"))

    available, reason = remote_memory_ready(session=session)

    assert available is False
    assert reason == READINESS_UNAVAILABLE_REASON


def test_readiness_probe_reports_the_reason_without_a_service_url(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("XAGENT_MEMORY_SERVICE_URL", raising=False)
    monkeypatch.setattr(remote_module, "_readiness_cache", None)
    session = FakeProbeSession(_response(200, {"status": "ok"}))

    available, reason = remote_memory_ready(session=session)

    assert available is False
    assert reason == READINESS_UNCONFIGURED_REASON
    assert session.calls == []


def test_readiness_answer_is_cached_until_refreshed(
    configured_remote_service: Any,
) -> None:
    ready = FakeProbeSession(_response(200, {"status": "ok"}))
    failing = FakeProbeSession(requests.exceptions.ConnectionError("refused"))

    assert remote_memory_ready(session=ready) == (True, None)
    # Inside the window the cached answer is reused: the service is not probed
    # again for every task startup.
    assert remote_memory_ready(session=failing) == (True, None)
    assert failing.calls == []
    # An explicit refresh always re-probes.
    assert remote_memory_ready(session=failing, refresh=True) == (
        False,
        READINESS_UNAVAILABLE_REASON,
    )
    assert len(failing.calls) == 1


# ---------------------------------------------------------------------------
# T14 §0.2: the agent-seat probe and the delegated identity override
# ---------------------------------------------------------------------------


@pytest.fixture
def configured_agent_seats(monkeypatch: Any) -> Any:
    from xagent.core.memory import agent_seats

    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", SERVICE_URL)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", SERVICE_KEY)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "1")
    registry = agent_seats.TrustedAgentSeatRegistry()
    monkeypatch.setattr(agent_seats, "_provider", registry)
    monkeypatch.setattr(agent_seats, "_provider_generation", 1)
    token = current_user_id.set(31)
    try:
        yield registry
    finally:
        current_user_id.reset(token)


def test_delegated_agent_identity_sets_and_resets_the_identity_header(
    configured_agent_seats: Any,
) -> None:
    configured_agent_seats.publish(
        service_url=SERVICE_URL,
        user_id=31,
        runtime_profile=SERVICE_PROFILE,
        agent_ids=[7],
        ttl_seconds=30,
        status="active",
    )
    session = FakeSession([_response(200, []), _response(200, []), _response(200, [])])
    store = RemoteMemoryStore(SERVICE_URL, SERVICE_KEY, session=session)
    store.search("parent")
    with delegated_agent_identity(7, store=store):
        store.search("seat")
    store.search("parent again")
    assert [call["headers"]["X-User-External-Id"] for call in session.calls] == [
        "31",
        "agent:7",
        "31",
    ]


def test_explicit_parent_mode_never_reads_seat_provider(
    monkeypatch: Any, configured_agent_seats: Any
) -> None:
    from xagent.core.memory import agent_seats

    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "0")
    monkeypatch.setattr(
        agent_seats,
        "_provider",
        lambda _request: pytest.fail("provider consulted in parent mode"),
    )
    store = RecordingStore(user_id=31)
    with delegated_agent_identity(7, store=store):
        assert store._headers()["X-User-External-Id"] == "31"
    assert store.session.calls == []


@pytest.mark.parametrize("identity", [None, "", " ", "23", 0, -1, False])
@pytest.mark.parametrize(
    "operation",
    [
        lambda s: s.add(_note()),
        lambda s: s.add(_note(content="[to:bob] body")),
        lambda s: s.add(_note(content="[share:company] body")),
        lambda s: s.remember(_note()),
        lambda s: s.get("id"),
        lambda s: s.update(_note()),
        lambda s: s.delete("id"),
        lambda s: s.search("query"),
        lambda s: s.list_all(),
        lambda s: s.get_stats(),
        lambda s: s.clear(),
        lambda s: s.inbox_list(),
        lambda s: s.inbox_read(["id"]),
        lambda s: s.delete_by_scope_dimension("client_application_id", "23"),
        lambda s: s.list_scope_dimension_values("client_application_id"),
    ],
)
def test_user_operations_require_identity_without_http(
    identity: Any, operation: Any
) -> None:
    store = RecordingStore(user_id=identity)
    assert store.session.calls == []  # Construction has no I/O.
    with pytest.raises(RemoteMemoryStoreIdentityError):
        operation(store)
    assert store.session.calls == []


def test_missing_identity_cannot_borrow_cached_seats_or_override_owner(
    configured_agent_seats: Any,
) -> None:
    configured_agent_seats.publish(
        service_url=SERVICE_URL,
        user_id=31,
        runtime_profile=SERVICE_PROFILE,
        agent_ids=[7],
        ttl_seconds=30,
        status="active",
    )
    token = current_user_id.set(None)
    calls = FakeSession([])
    store = RemoteMemoryStore(SERVICE_URL, SERVICE_KEY, session=calls)
    try:
        with explicit_memory_owner(99), pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("no identity")
        with pytest.raises(RemoteMemoryStoreIdentityError):
            registered_agent_seats()
        with (
            pytest.raises(RemoteMemoryStoreIdentityError),
            delegated_agent_identity(7, store=store),
        ):
            pytest.fail("missing identity entered a seat")
    finally:
        current_user_id.reset(token)
    assert calls.calls == []


def test_seat_provider_is_scoped_to_identity_and_service(
    configured_agent_seats: Any, monkeypatch: Any
) -> None:
    configured_agent_seats.publish(
        service_url=SERVICE_URL,
        user_id=31,
        runtime_profile=SERVICE_PROFILE,
        agent_ids=[7],
        ttl_seconds=30,
        status="active",
    )
    assert registered_agent_seats()[0] == frozenset({"7"})
    assert registered_agent_seats(user_id=32)[0] is None
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://other-service")
    assert registered_agent_seats()[0] is None


def test_missing_and_expired_mapping_never_triggers_delegation_http(
    configured_agent_seats: Any, monkeypatch: Any
) -> None:
    from xagent.core.memory import agent_seats

    clock = [100.0]
    monkeypatch.setattr(agent_seats.time, "monotonic", lambda: clock[0])
    store = RecordingStore(user_id=31)
    with (
        pytest.raises(RemoteMemoryStoreIdentityError),
        delegated_agent_identity(7, store=store),
    ):
        pytest.fail("missing map entered")
    configured_agent_seats.publish(
        service_url=SERVICE_URL,
        user_id=31,
        runtime_profile=SERVICE_PROFILE,
        agent_ids=[7],
        ttl_seconds=30,
        status="active",
    )
    clock[0] += 31
    with (
        pytest.raises(RemoteMemoryStoreIdentityError),
        delegated_agent_identity(7, store=store),
    ):
        pytest.fail("expired map entered")
    assert store.session.calls == []
