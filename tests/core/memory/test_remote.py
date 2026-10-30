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
    AGENT_SEATS_PATH,
    AGENT_SEATS_PROBE_FAILED_REASON,
    NOT_FOUND_OR_DENIED_ERROR,
    READINESS_UNAVAILABLE_REASON,
    READINESS_UNCONFIGURED_REASON,
    READY_PATH,
    SCOPE_DIMENSIONS_FILTER_KEY,
    WHOAMI_PATH,
    RemoteMemoryStore,
    RemoteMemoryStoreAuthenticationError,
    RemoteMemoryStoreConfigurationError,
    RemoteMemoryStoreError,
    RemoteMemoryStoreIncompleteError,
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

    def __init__(self, *responses: Any, user_id: Optional[int] = None) -> None:
        self.session = FakeSession(list(responses))
        super().__init__(
            SERVICE_URL,
            SERVICE_KEY,
            session=self.session,  # type: ignore[arg-type]
            user_id_provider=lambda: user_id,
        )


@pytest.fixture(autouse=True)
def _isolated_remote_diagnostics():
    """Keep the 401-classification cache and the health signal test-local."""
    remote_module._key_validity_cache.clear()
    remote_module._upstream_health.update(
        {
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
    )
    yield
    remote_module._key_validity_cache.clear()


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


def test_remember_surfaces_the_service_degraded_flag() -> None:
    """The service's ``degraded`` flag is not dropped (X-04)."""
    store = RecordingStore(
        _response(
            200,
            {
                "action": "merge",
                "id": "n",
                "parent_id": "p",
                "degraded": True,
            },
        )
    )

    response = store.remember(_note())

    assert response.metadata == {
        "action": "merge",
        "parent_id": "p",
        "degraded": True,
    }


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


def test_by_id_access_not_narrowed_by_dimensions(scope: Any) -> None:
    """Dimensions are stamp-only: a dimensioned scope does not gate by id."""
    store = RecordingStore(
        _response(
            200,
            {"id": "note-1", "content": "x", "metadata": {"execution_scope_a": "9"}},
        )
    )

    response = store.get("note-1")

    assert response.success is True
    assert [call["method"] for call in store.session.calls] == ["GET"]


def test_strict_with_dimensions_does_not_gate_by_id() -> None:
    """strict + own dimensions: the one-way rule needs a dimension-less scope."""
    token = set_execution_scope(
        ExecutionScope(memory_dimensions={"task": "7"}, strict_memory_isolation=True)
    )
    try:
        store = RecordingStore(
            _response(
                200,
                {
                    "id": "note-1",
                    "content": "x",
                    "metadata": {"execution_scope_a": "1"},
                },
            )
        )

        response = store.get("note-1")
    finally:
        reset_execution_scope(token)

    assert response.success is True


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


def test_search_without_a_threshold_sends_the_measured_distance_ceiling() -> None:
    """No caller threshold must not travel as ``null`` (= most of the store).

    ``similarity_threshold`` is a cosine distance ceiling upstream, so the
    fallback is the measured 0.40 (similarity >= 0.60), not the 1.5 placeholder.
    """
    store = RecordingStore(_response(200, []))

    store.search("hello")

    call = store.session.calls[-1]
    assert call["url"] == f"{SERVICE_URL}/v1/memory/search"
    assert call["json"]["similarity_threshold"] == pytest.approx(0.40)
    assert remote_module.DEFAULT_SEARCH_SIMILARITY_THRESHOLD == pytest.approx(0.40)


def test_search_default_does_not_merge_scope_dimensions(scope: Any) -> None:
    """Dimensions are stamp-only: without the opt-in key they never filter.

    Auto-merging would blind cross-task recall and store_memory dedup the
    moment a task scope carries dimensions.
    """
    store = RecordingStore(_response(200, []))

    store.search("hello", filters={"category": "general"})

    assert store.session.calls[-1]["json"]["filters"] == {"category": "general"}


def test_search_opt_in_merges_scope_dimensions(scope: Any) -> None:
    """The opt-in key merges the active scope's dimensions and is stripped."""
    store = RecordingStore(_response(200, []))
    filters = {"category": "general", SCOPE_DIMENSIONS_FILTER_KEY: True}

    store.search("hello", filters=filters)

    assert store.session.calls[-1]["json"]["filters"] == {
        "category": "general",
        "metadata": {"execution_scope_client_application_id": "42"},
    }
    # The caller's dict is never mutated (the reserved key stays put).
    assert filters == {
        "category": "general",
        SCOPE_DIMENSIONS_FILTER_KEY: True,
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
    store = RecordingStore(
        _response(200, {"memories": [{"id": "n1", "content": "x"}], "truncated": False})
    )

    notes = store.list_all({"category": "general"})

    assert [note.id for note in notes] == ["n1"]
    call = store.session.calls[-1]
    assert call["method"] == "POST"
    assert call["url"] == f"{SERVICE_URL}/v1/memory/list"
    assert call["json"] == {"filters": {"category": "general"}}


def test_list_all_opt_in_merges_scope_dimensions(scope: Any) -> None:
    store = RecordingStore(_response(200, {"results": [], "truncated": False}))

    store.list_all({SCOPE_DIMENSIONS_FILTER_KEY: True})

    assert store.session.calls[-1]["json"]["filters"] == {
        "metadata": {"execution_scope_client_application_id": "42"}
    }


@pytest.mark.parametrize("count", [0, 9999, 10000, 10001])
@pytest.mark.parametrize("filters", [None, {"category": "general"}])
def test_list_all_requires_complete_results(count: int, filters: Any) -> None:
    items = [{"id": str(i), "content": "x"} for i in range(min(count, 10000))]
    store = RecordingStore(_response(200, {"notes": items, "truncated": count > 10000}))
    if count > 10000:
        with pytest.raises(RemoteMemoryStoreIncompleteError):
            store.list_all(filters)
    else:
        assert len(store.list_all(filters)) == count
    assert len(store.session.calls) == 1
    assert store.session.calls[0]["json"] == {"filters": filters}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"notes": []},
        {"notes": [], "truncated": 0},
        {"notes": [], "truncated": "false"},
        {"notes": [], "truncated": None},
    ],
)
def test_list_all_rejects_unknown_completeness(payload: Any) -> None:
    store = RecordingStore(_response(200, payload))
    with pytest.raises(RemoteMemoryStoreError, match="completeness flag"):
        store.list_all()


def test_bounded_search_and_inbox_keep_their_envelope_contract() -> None:
    store = RecordingStore(
        _response(200, {"notes": [{"id": "a"}], "truncated": True}),
        _response(200, {"notes": [{"id": "b"}]}),
    )
    assert [note.id for note in store.search("query")] == ["a"]
    assert [note.id for note in store.inbox_list()] == ["b"]


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


# ---------------------------------------------------------------------------
# 缺陷一: a 401 must be classified from observable evidence, never blamed on
# the credential by default; transport failures must say "upstream
# unreachable", not "credential"; other failures carry status/code/message.
# ---------------------------------------------------------------------------

_GENERIC_401 = {"error": {"code": "unauthorized", "message": "unauthorized"}}


def test_401_with_credential_wording_blames_the_key_without_probing() -> None:
    store = RecordingStore(
        _response(
            401, {"error": {"code": "unauthorized", "message": "invalid api key"}}
        )
    )

    with pytest.raises(RemoteMemoryStoreAuthenticationError) as error:
        store.search("hello")

    text = str(error.value)
    assert "拒绝了服务凭据" in text
    assert "invalid api key" in text
    assert "HTTP 401" in text
    # an explicit credential error needs no diagnostic probe
    assert len(store.session.calls) == 1


def test_401_with_unknown_external_id_reports_identity_not_provisioned() -> None:
    store = RecordingStore(
        _response(401, _GENERIC_401),
        _response(200, {"identity": {"external_id": "1"}}),
        user_id=2,
    )

    with pytest.raises(RemoteMemoryStoreAuthenticationError) as error:
        store.search("hello")

    text = str(error.value)
    assert "身份未开通" in text
    assert "external_id=2" in text
    assert "memsrv org" in text
    # the bare-key probe separates the two 401 causes
    probe = store.session.calls[1]
    assert probe["url"] == f"{SERVICE_URL}{WHOAMI_PATH}"
    assert probe["headers"]["Authorization"] == f"Bearer {SERVICE_KEY}"
    assert "X-User-External-Id" not in probe["headers"]


def test_401_blames_the_key_when_the_bare_key_is_also_rejected() -> None:
    store = RecordingStore(
        _response(401, _GENERIC_401),
        _response(401, _GENERIC_401),
        user_id=2,
    )

    with pytest.raises(RemoteMemoryStoreAuthenticationError) as error:
        store.search("hello")

    text = str(error.value)
    assert "拒绝了服务凭据" in text
    assert "探测同样 401" in text


def test_401_names_both_hypotheses_when_the_probe_cannot_decide() -> None:
    store = RecordingStore(
        _response(401, _GENERIC_401),
        requests.exceptions.ConnectionError("refused"),
        user_id=2,
    )

    with pytest.raises(RemoteMemoryStoreAuthenticationError) as error:
        store.search("hello")

    text = str(error.value)
    assert "未能完成" in text
    assert "key 无效" in text
    assert "external_id=2" in text


def test_401_without_an_identity_header_reports_the_key_itself() -> None:
    store = RecordingStore(_response(401, _GENERIC_401))

    with pytest.raises(RemoteMemoryStoreAuthenticationError) as error:
        store.search("hello")

    text = str(error.value)
    assert "未携带 external_id" in text
    assert len(store.session.calls) == 1


def test_401_classification_reuses_the_cached_key_verdict() -> None:
    store = RecordingStore(
        _response(401, _GENERIC_401),
        _response(200, {"identity": {"external_id": "1"}}),
        _response(401, _GENERIC_401),
        user_id=2,
    )

    with pytest.raises(RemoteMemoryStoreAuthenticationError) as first:
        store.search("hello")
    with pytest.raises(RemoteMemoryStoreAuthenticationError) as second:
        store.list_all()

    assert "身份未开通" in str(first.value)
    assert "身份未开通" in str(second.value)
    # three calls total: 401, one probe, 401 -- the verdict is cached
    assert len(store.session.calls) == 3


def test_timeout_reports_an_unreachable_upstream_not_a_credential() -> None:
    store = RecordingStore(requests.exceptions.Timeout("slow"))

    with pytest.raises(RemoteMemoryStoreError) as error:
        store.search("hello")

    text = str(error.value)
    assert "unreachable" in text
    assert "not a credential rejection" in text
    assert "timed out" in text


def test_connection_failure_reports_an_unreachable_upstream_not_a_credential() -> (
    None
):
    store = RecordingStore(requests.exceptions.ConnectionError("refused"))

    with pytest.raises(RemoteMemoryStoreError) as error:
        store.list_all()

    text = str(error.value)
    assert "unreachable" in text
    assert "not a credential rejection" in text
    assert "refused" in text


def test_non_401_failure_carries_status_code_and_upstream_message() -> None:
    store = RecordingStore(
        _response(500, {"error": {"code": "internal", "message": "boom"}})
    )

    with pytest.raises(RemoteMemoryStoreError) as error:
        store.search("hello")

    text = str(error.value)
    assert "HTTP 500" in text
    assert "code=internal" in text
    assert "boom" in text
    assert error.value.status_code == 500


# ---------------------------------------------------------------------------
# 缺陷二: the upstream health signal every answered/failed call records, and
# the ready/degraded/unavailable derivation store-info publishes from it.
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


def test_answered_calls_record_the_upstream_health_signal() -> None:
    store = RecordingStore(_response(200, {"results": [], "truncated": False}))

    store.list_all()

    health = remote_module.remote_upstream_health()
    assert health["last_event_kind"] == "answer"
    assert health["last_status"] == 200
    assert health["last_success_at"] is not None
    assert health["consecutive_transport_failures"] == 0


def test_transport_failures_record_the_upstream_health_signal() -> None:
    store = RecordingStore(requests.exceptions.ConnectionError("refused"))

    with pytest.raises(RemoteMemoryStoreError):
        store.list_all()

    health = remote_module.remote_upstream_health()
    assert health["last_event_kind"] == "transport_error"
    assert health["consecutive_transport_failures"] == 1
    assert "连接失败" in health["last_failure_reason"]


def test_upstream_report_is_ready_after_an_answered_call() -> None:
    health = _health(
        last_event_at=1761710000.0,
        last_event_kind="answer",
        last_status=200,
        last_path="/v1/memory/list",
        last_success_at=1761710000.0,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(remote_module, "_upstream_health", health)
        state, detail = remote_module.remote_upstream_report()

    assert state == "ready"
    assert "HTTP 200" in detail
    assert "/v1/memory/list" in detail


def test_upstream_report_degrades_on_a_5xx_answer() -> None:
    health = _health(
        last_event_at=1761710000.0,
        last_event_kind="answer",
        last_status=503,
        last_path="/v1/memory/list",
        last_error_code="resource_exhausted",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(remote_module, "_upstream_health", health)
        state, detail = remote_module.remote_upstream_report()

    assert state == "degraded"
    assert "HTTP 503" in detail
    assert "code=resource_exhausted" in detail


def test_upstream_report_degrades_on_a_transport_failure_after_success() -> None:
    health = _health(
        last_event_at=1761710000.0,
        last_event_kind="transport_error",
        last_failure_at=1761710000.0,
        last_failure_reason="读超时 2.0s",
        last_path="/v1/memory/search",
        last_success_at=1761709000.0,
        consecutive_transport_failures=2,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(remote_module, "_upstream_health", health)
        state, detail = remote_module.remote_upstream_report()

    assert state == "degraded"
    assert "不可达" in detail
    assert "/v1/memory/search" in detail


def test_upstream_report_is_unavailable_without_any_successful_call() -> None:
    health = _health(
        last_event_at=1761710000.0,
        last_event_kind="transport_error",
        last_failure_at=1761710000.0,
        last_failure_reason="连接失败（ConnectionError）",
        last_path="/v1/memory/list",
        consecutive_transport_failures=1,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(remote_module, "_upstream_health", health)
        state, detail = remote_module.remote_upstream_report()

    assert state == "unavailable"
    assert "不可达" in detail


def test_upstream_report_falls_back_to_the_cached_readiness_probe() -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(remote_module, "_upstream_health", _health())
        patch.setattr(
            remote_module, "remote_memory_ready", lambda **_kwargs: (True, None)
        )
        state, detail = remote_module.remote_upstream_report()
    assert state == "ready"
    assert "/readyz" in detail

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(remote_module, "_upstream_health", _health())
        patch.setattr(
            remote_module,
            "remote_memory_ready",
            lambda **_kwargs: (False, READINESS_UNAVAILABLE_REASON),
        )
        state, detail = remote_module.remote_upstream_report()
    assert state == "unavailable"
    assert READINESS_UNAVAILABLE_REASON in detail


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


def test_identity_header_is_absent_without_a_user() -> None:
    session = FakeSession([_response(200, [])])
    token = current_user_id.set(None)
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
    assert "X-User-External-Id" not in headers


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
    """Configure the service and clear the process seat cache."""
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", SERVICE_URL)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", SERVICE_KEY)
    monkeypatch.delenv("XAGENT_MEMORY_AGENT_SEATS", raising=False)
    monkeypatch.setattr(remote_module, "_agent_seats_cache", None)
    yield
    monkeypatch.setattr(remote_module, "_agent_seats_cache", None)


def test_agent_seat_probe_lists_the_registered_refs(
    configured_agent_seats: Any,
) -> None:
    session = FakeSession(
        [
            _response(
                200,
                {
                    "users": [
                        {"id": 61, "kind": "agent", "agent_ref": "77030"},
                        {"id": 34, "kind": "person", "agent_ref": None},
                        {"id": 62, "kind": "agent"},  # no ref -> not addressable
                        "not-a-user",
                    ]
                },
            )
        ]
    )

    seats, reason = registered_agent_seats(session=session)

    assert seats == frozenset({"77030"})
    assert reason is None
    call = session.calls[-1]
    assert call["url"] == f"{SERVICE_URL}{AGENT_SEATS_PATH}"
    assert call["params"] == {"kind": "agent"}
    assert call["headers"]["Authorization"] == f"Bearer {SERVICE_KEY}"
    assert call["timeout"] == (1.0, 2.0)


def test_agent_seat_answer_is_cached_until_refreshed(
    configured_agent_seats: Any,
) -> None:
    probe = FakeSession(
        [_response(200, {"users": [{"kind": "agent", "agent_ref": "7"}]})]
    )
    failing = FakeSession([requests.exceptions.ConnectionError("refused")])

    assert registered_agent_seats(session=probe) == (frozenset({"7"}), None)
    # Inside the window the cached answer is reused: no probe per delegation.
    assert registered_agent_seats(session=failing) == (frozenset({"7"}), None)
    assert failing.calls == []
    # A refresh re-probes, and a failure is reported instead of raised.
    seats, reason = registered_agent_seats(session=failing, refresh=True)
    assert seats is None
    assert reason == AGENT_SEATS_PROBE_FAILED_REASON
    assert len(failing.calls) == 1


@pytest.mark.parametrize(
    "answer",
    [
        _response(403, {"error": {"code": "forbidden"}}),
        _response(500, {}),
        requests.exceptions.ConnectionError("refused"),
        _response(200, {"users": "not-a-list"}),
    ],
)
def test_agent_seat_probe_failure_is_reported_not_raised(
    configured_agent_seats: Any, answer: Any
) -> None:
    session = FakeSession([answer])

    seats, reason = registered_agent_seats(session=session)

    assert seats is None
    assert reason == AGENT_SEATS_PROBE_FAILED_REASON


def test_agent_seat_probe_without_a_service_url(monkeypatch: Any) -> None:
    monkeypatch.delenv("XAGENT_MEMORY_SERVICE_URL", raising=False)
    monkeypatch.setattr(remote_module, "_agent_seats_cache", None)
    session = FakeSession([_response(200, {"users": []})])

    seats, reason = registered_agent_seats(session=session)

    assert seats is None
    assert reason == READINESS_UNCONFIGURED_REASON
    assert session.calls == []


def test_delegated_agent_identity_sets_and_resets_the_identity_header(
    configured_agent_seats: Any,
) -> None:
    probe = FakeSession(
        [_response(200, {"users": [{"kind": "agent", "agent_ref": "7"}]})]
    )
    calls = FakeSession([_response(200, []), _response(200, []), _response(200, [])])
    store = RemoteMemoryStore(
        SERVICE_URL,
        SERVICE_KEY,
        session=calls,  # type: ignore[arg-type]
    )
    token = current_user_id.set(31)
    try:
        store.search("parent identity")
        with delegated_agent_identity(7, store=store, session=probe):
            store.search("child identity")
        store.search("back to the parent")
    finally:
        current_user_id.reset(token)

    assert calls.calls[0]["headers"]["X-User-External-Id"] == "31"
    assert calls.calls[1]["headers"]["X-User-External-Id"] == "agent:7"
    assert calls.calls[2]["headers"]["X-User-External-Id"] == "31"
    assert len(probe.calls) == 1, "the seat probe runs once per cache window"


def test_delegated_agent_identity_is_a_noop_for_unregistered_agents_and_other_stores(
    configured_agent_seats: Any,
) -> None:
    probe = FakeSession(
        [_response(200, {"users": [{"kind": "agent", "agent_ref": "7"}]})]
    )
    calls = FakeSession([_response(200, []), _response(200, [])])
    store = RemoteMemoryStore(
        SERVICE_URL,
        SERVICE_KEY,
        session=calls,  # type: ignore[arg-type]
    )
    token = current_user_id.set(31)
    try:
        with delegated_agent_identity(8, store=store, session=probe):
            store.search("unregistered agent")
        with delegated_agent_identity(7, store=object(), session=probe):
            store.search("not the remote adapter")
    finally:
        current_user_id.reset(token)

    assert calls.calls[0]["headers"]["X-User-External-Id"] == "31"
    assert calls.calls[1]["headers"]["X-User-External-Id"] == "31"


def test_delegated_agent_identity_falls_back_when_the_probe_fails(
    configured_agent_seats: Any,
) -> None:
    probe = FakeSession([requests.exceptions.ConnectionError("refused")])
    calls = FakeSession([_response(200, [])])
    store = RemoteMemoryStore(
        SERVICE_URL,
        SERVICE_KEY,
        session=calls,  # type: ignore[arg-type]
    )
    token = current_user_id.set(31)
    try:
        with delegated_agent_identity(7, store=store, session=probe):
            store.search("service down")
    finally:
        current_user_id.reset(token)

    assert calls.calls[0]["headers"]["X-User-External-Id"] == "31", (
        "a failed probe must never block the delegation nor invent a seat"
    )


def test_delegated_agent_identity_can_be_switched_off(
    monkeypatch: Any, configured_agent_seats: Any
) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "0")
    probe = FakeSession(
        [_response(200, {"users": [{"kind": "agent", "agent_ref": "7"}]})]
    )
    calls = FakeSession([_response(200, [])])
    store = RemoteMemoryStore(
        SERVICE_URL,
        SERVICE_KEY,
        session=calls,  # type: ignore[arg-type]
    )
    token = current_user_id.set(31)
    try:
        with delegated_agent_identity(7, store=store, session=probe):
            store.search("switch off")
    finally:
        current_user_id.reset(token)

    assert calls.calls[0]["headers"]["X-User-External-Id"] == "31"
    assert probe.calls == [], "a disabled switch must not even probe"


@pytest.mark.parametrize("truncated", [False, True])
def test_memsrv_list_envelope_keeps_scope_and_completeness(
    scope: Any, truncated: bool
) -> None:
    envelope = {
        "results": [
            {
                "id": "scoped",
                "content": "body",
                "metadata": {"execution_scope_client_application_id": "42"},
            }
        ],
        "limit": 10000,
        "truncated": truncated,
    }
    store = RecordingStore(_response(200, envelope))
    if truncated:
        with pytest.raises(RemoteMemoryStoreIncompleteError):
            store.list_all({"category": "general", SCOPE_DIMENSIONS_FILTER_KEY: True})
    else:
        notes = store.list_all(
            {"category": "general", SCOPE_DIMENSIONS_FILTER_KEY: True}
        )
        assert notes[0].id == "scoped"
        assert notes[0].metadata["execution_scope_client_application_id"] == "42"
    assert store.session.calls[0]["json"] == {
        "filters": {
            "category": "general",
            "metadata": {"execution_scope_client_application_id": "42"},
        }
    }
