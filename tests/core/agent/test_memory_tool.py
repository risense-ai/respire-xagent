"""Tests for the model-invocable ``store_memory`` tool."""

import asyncio
import json
from datetime import datetime
from typing import Any

import pytest
import requests

from xagent.core.agent.context.memory_tool import (
    READBACK_REJECTED_MESSAGE,
    RECEIPT_FAILURE_UNKNOWN_MESSAGE,
    UNVERIFIED_RECEIPT_MESSAGE,
    CheckInboxTool,
    DeleteMemoryTool,
    DeliverMemoryTool,
    SearchMemoryArgs,
    SearchMemoryTool,
    StoreMemoryTool,
    UpdateMemoryTool,
    build_memory_tools,
    build_store_memory_tool,
)
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.user_context import current_user_id
from xagent.core.memory.remote import (
    RemoteMemoryStore,
    RemoteMemoryStoreAuthenticationError,
    RemoteMemoryStoreError,
)
from xagent.web.memory_lifecycle import MemoryLifecycleState, MemoryLifecycleStatus
from xagent.web.revocable_memory_store import RevocableMemoryStore
from xagent.web.user_isolated_memory import UserIsolatedMemoryStore

SERVICE_URL = "http://memsrv:8080"
SERVICE_KEY = "svc-key"


class RecordingMemoryStore:
    def __init__(
        self,
        *,
        search_results: list[Any] | None = None,
        add_success: bool = True,
    ) -> None:
        self.search_results = search_results or []
        self.add_success = add_success
        self.searches: list[dict[str, Any]] = []
        self.added: list[MemoryNote] = []

    def search(self, **kwargs: Any) -> list[Any]:
        self.searches.append(kwargs)
        return list(self.search_results)

    def add(self, note: MemoryNote) -> MemoryResponse:
        self.added.append(note)
        if not self.add_success:
            return MemoryResponse(success=False, error="boom")
        return MemoryResponse(success=True, memory_id=f"mem-{len(self.added)}")


class TraceEventRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "data": data or {},
            }
        )
        return str(len(self.events))


class FakeRuntime:
    def __init__(self, tracer: Any | None = None) -> None:
        self.tracer = tracer
        self.execution_id = "task-1"
        self.active_react_step_id = "react_step_1"


@pytest.mark.asyncio
async def test_store_memory_adds_note_with_metadata() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="Fix the deploy pipeline")

    result = await tool.execute(
        content="User prefers reports in Chinese.", kind="user_preference"
    )

    assert result == {
        "success": True,
        "stored": True,
        "action": "create",
        "memory_id": "mem-1",
        "message": "Stored as a new memory (mem-1).",
    }
    note = store.added[0]
    assert note.content == "User prefers reports in Chinese."
    assert note.category == "react_memory"
    # The user's request text is conversation content and must not outlive a
    # purged conversation inside the memory store (#2753).
    assert note.metadata == {"kind": "user_preference", "source": "store_memory"}
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_merges_into_a_similar_note() -> None:
    """A similar hit is folded into the existing note, never dropped.

    The old behaviour skipped the write ("stored": false) and lost the new
    fact; the judged path keeps both by updating the note in place.
    """
    store = CrudMemoryStore({"mem-1": _note("mem-1", "Same insight.")})
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="Same insight, refined.")

    assert result["success"] is True
    assert result["stored"] is True
    assert result["action"] == "update"
    assert store.updated[0].id == "mem-1"
    assert "Same insight, refined." in str(store.updated[0].content)
    assert store.searches[0]["filters"] == {"category": "react_memory"}
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_enforces_per_run_quota() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="task", max_stores=2)

    assert (await tool.execute(content="First insight."))["success"] is True
    assert (await tool.execute(content="Second insight."))["success"] is True
    result = await tool.execute(content="Third insight.")

    assert result["success"] is False
    assert "limit" in result["error"]
    assert len(store.added) == 2


@pytest.mark.asyncio
async def test_store_memory_serializes_concurrent_quota_checks() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="task", max_stores=1)

    results = await asyncio.gather(
        tool.execute(content="First concurrent insight."),
        tool.execute(content="Second concurrent insight."),
    )

    assert sum(result.get("stored") is True for result in results) == 1
    assert sum("limit" in result.get("error", "") for result in results) == 1
    assert len(store.added) == 1
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_rejects_empty_content() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="   ")

    assert result["success"] is False
    assert store.added == []


@pytest.mark.asyncio
async def test_store_memory_normalizes_unknown_kind() -> None:
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="An insight.", kind="made_up_kind")

    assert result["success"] is True
    assert store.added[0].metadata["kind"] == "domain_insight"


@pytest.mark.asyncio
async def test_store_memory_reports_add_failure() -> None:
    store = RecordingMemoryStore(add_success=False)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="An insight.")

    assert result["success"] is False
    assert tool.stored_count == 0


@pytest.mark.asyncio
async def test_store_memory_emits_store_trace_events() -> None:
    tracer = TraceEventRecorder()
    store = RecordingMemoryStore()
    tool = StoreMemoryTool(
        memory_store=store,
        task="task",
        runtime=FakeRuntime(tracer=tracer),
    )

    await tool.execute(content="An insight.")

    event_types = [event["event_type"] for event in tracer.events]
    assert event_types == ["task_start_memory_store", "task_end_memory_store"]
    assert all(event["task_id"] == "task-1" for event in tracer.events)
    assert tracer.events[1]["data"]["storage_success"] is True
    assert tracer.events[1]["data"]["memory_id"] == "mem-1"


@pytest.mark.asyncio
async def test_store_memory_stores_when_dedup_search_fails() -> None:
    class BrokenSearchStore(RecordingMemoryStore):
        def search(self, **kwargs: Any) -> list[Any]:
            raise RuntimeError("search unavailable")

    store = BrokenSearchStore()
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="An insight.")

    assert result["success"] is True
    assert len(store.added) == 1


def test_build_store_memory_tool_requires_store() -> None:
    assert build_store_memory_tool(memory_store=None, task="task") is None
    tool = build_store_memory_tool(memory_store=RecordingMemoryStore(), task="task")
    assert isinstance(tool, StoreMemoryTool)
    assert tool.name == "store_memory"


class CrudMemoryStore(RecordingMemoryStore):
    """Recording store with get/update/delete for the CRUD tools."""

    def __init__(self, notes: dict[str, MemoryNote] | None = None) -> None:
        super().__init__()
        self.notes = dict(notes or {})
        self.updated: list[MemoryNote] = []
        self.deleted: list[str] = []

    def search(self, **kwargs: Any) -> list[Any]:
        self.searches.append(kwargs)
        return list(self.notes.values())

    def get(self, note_id: str) -> MemoryResponse:
        note = self.notes.get(note_id)
        if note is None:
            return MemoryResponse(success=False, error="Note not found")
        return MemoryResponse(success=True, memory_id=note_id, content=note)

    def update(self, note: MemoryNote) -> MemoryResponse:
        self.updated.append(note)
        self.notes[note.id] = note
        return MemoryResponse(success=True, memory_id=note.id)

    def delete(self, note_id: str) -> MemoryResponse:
        if note_id not in self.notes:
            return MemoryResponse(success=False, error="Note not found")
        del self.notes[note_id]
        self.deleted.append(note_id)
        return MemoryResponse(success=True, memory_id=note_id)


def _note(note_id: str, content: str) -> MemoryNote:
    return MemoryNote(id=note_id, content=content, category="react_memory")


@pytest.mark.asyncio
async def test_search_memory_returns_ids_and_content() -> None:
    store = CrudMemoryStore({"mem-1": _note("mem-1", "User prefers Chinese.")})
    tool = SearchMemoryTool(memory_store=store)

    result = await tool.execute(query="preferences")

    assert result["success"] is True
    assert result["count"] == 1
    assert result["memories"] == [
        {
            "id": "mem-1",
            "content": "User prefers Chinese.",
            "category": "react_memory",
            "space_kind": None,
            "space_id": None,
        }
    ]
    assert store.searches[0]["query"] == "preferences"


@pytest.mark.asyncio
async def test_search_memory_carries_the_owner_of_each_result() -> None:
    """X-07: an organisation-scope search must say whose note each hit is.

    The service already limits what a regular user can read, so this is
    attribution, not an access change. A note without a stored owner stays
    without one -- no owner is invented for it.
    """
    store = CrudMemoryStore(
        {
            "mem-2": MemoryNote(
                id="mem-2",
                content="2 号的报销额度是 8800 元。",
                category="general",
                metadata={"user_id": 2},
            ),
            "mem-3": MemoryNote(
                id="mem-3",
                content="3 号的车牌号是 京A9999。",
                category="general",
                metadata={"user_id": 3},
            ),
            "mem-old": MemoryNote(
                id="mem-old",
                content="没有归属的旧条目。",
                category="general",
                metadata={},
            ),
        }
    )
    tool = SearchMemoryTool(memory_store=store)

    result = await tool.execute(query="谁的报销额度和车牌号")

    by_id = {item["id"]: item for item in result["memories"]}
    assert by_id["mem-2"]["owner_user_id"] == 2
    assert by_id["mem-3"]["owner_user_id"] == 3
    assert "owner_user_id" not in by_id["mem-old"]


@pytest.mark.asyncio
async def test_search_memory_keeps_different_owners_apart_in_one_answer() -> None:
    """One query hitting several people returns each fact with its own owner."""
    store = CrudMemoryStore(
        {
            "mem-2": MemoryNote(
                id="mem-2",
                content="2 号的停车月卡登记用 京A8888。",
                category="general",
                metadata={"user_id": 2},
            ),
            "mem-3": MemoryNote(
                id="mem-3",
                content="3 号的停车月卡登记用 京B0001。",
                category="general",
                metadata={"user_id": 3},
            ),
        }
    )
    tool = SearchMemoryTool(memory_store=store)

    result = await tool.execute(query="停车月卡登记的是哪个车牌")

    assert result["count"] == 2
    owners_by_content = {
        item["content"]: item["owner_user_id"] for item in result["memories"]
    }
    assert owners_by_content == {
        "2 号的停车月卡登记用 京A8888。": 2,
        "3 号的停车月卡登记用 京B0001。": 3,
    }


@pytest.mark.asyncio
async def test_search_memory_rejects_empty_query_and_emits_trace() -> None:
    tracer = TraceEventRecorder()
    store = CrudMemoryStore()
    tool = SearchMemoryTool(memory_store=store, runtime=FakeRuntime(tracer=tracer))

    empty = await tool.execute(query="  ")
    ok = await tool.execute(query="anything")

    assert empty["success"] is False
    assert ok["success"] is True
    event_types = [event["event_type"] for event in tracer.events]
    assert event_types == [
        "task_start_memory_retrieve",
        "task_end_memory_retrieve",
    ]
    assert tracer.events[-1]["data"]["success"] is True


@pytest.mark.asyncio
async def test_search_memory_reports_backend_failure() -> None:
    class BrokenSearchStore(CrudMemoryStore):
        def search(self, **kwargs: Any) -> list[Any]:
            raise RuntimeError("backend unavailable")

    tracer = TraceEventRecorder()
    tool = SearchMemoryTool(
        memory_store=BrokenSearchStore(), runtime=FakeRuntime(tracer=tracer)
    )

    result = await tool.execute(query="preferences")

    # P1-4: the raised error's own message is surfaced to the model instead of
    # a generic string that hides what failed.
    assert result == {"success": False, "error": "backend unavailable"}
    assert tracer.events[-1]["data"]["success"] is False
    assert tracer.events[-1]["data"]["found"] is False


@pytest.mark.asyncio
async def test_search_memory_reports_terminal_lancedb_failure(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenConnection:
        def open_table(self, _: str) -> Any:
            raise RuntimeError("memory table unavailable")

    base_store = LanceDBMemoryStore(
        db_dir=str(tmp_path),
        collection_name="terminal_failure",
    )
    monkeypatch.setattr(
        base_store._vector_store,
        "get_raw_connection",
        lambda: BrokenConnection(),
    )
    tool = SearchMemoryTool(
        memory_store=UserIsolatedMemoryStore(base_store),
    )

    result = await tool.execute(query="preferences")

    # P1-4: the propagated message is the user-facing error (str(e), no
    # traceback), so the exact text now follows the exception.
    assert result["success"] is False
    assert "memory table unavailable" in result["error"]


@pytest.mark.asyncio
async def test_update_memory_replaces_content() -> None:
    store = CrudMemoryStore({"mem-1": _note("mem-1", "Old fact.")})
    tool = UpdateMemoryTool(memory_store=store)

    result = await tool.execute(memory_id="mem-1", content="Corrected fact.")

    assert result == {"success": True, "memory_id": "mem-1"}
    assert store.updated[0].content == "Corrected fact."
    # No request text is stamped onto the rewritten note (#2753).
    assert store.updated[0].metadata == {}


@pytest.mark.asyncio
async def test_update_memory_initializes_missing_metadata() -> None:
    note = _note("mem-1", "Old fact.")
    note.metadata = None  # type: ignore[assignment]
    store = CrudMemoryStore({"mem-1": note})
    tool = UpdateMemoryTool(memory_store=store)

    result = await tool.execute(memory_id="mem-1", content="Corrected fact.")

    assert result == {"success": True, "memory_id": "mem-1"}
    assert store.updated[0].metadata == {}


@pytest.mark.asyncio
async def test_update_memory_reports_missing_note_and_bad_args() -> None:
    store = CrudMemoryStore()
    tool = UpdateMemoryTool(memory_store=store)

    missing = await tool.execute(memory_id="nope", content="New text.")
    bad = await tool.execute(memory_id="", content="")

    assert missing["success"] is False
    assert "not found" in missing["error"]
    assert bad["success"] is False
    assert store.updated == []


@pytest.mark.asyncio
async def test_delete_memory_removes_note() -> None:
    store = CrudMemoryStore({"mem-1": _note("mem-1", "Wrong fact.")})
    tool = DeleteMemoryTool(memory_store=store)

    result = await tool.execute(memory_id="mem-1")
    missing = await tool.execute(memory_id="mem-1")

    assert result == {"success": True, "memory_id": "mem-1"}
    assert store.deleted == ["mem-1"]
    assert missing["success"] is False


def test_build_memory_tools_composition() -> None:
    assert build_memory_tools(memory_store=None, task="task") == []
    tools = build_memory_tools(memory_store=CrudMemoryStore(), task="task")
    assert [tool.name for tool in tools] == [
        "store_memory",
        "search_memory",
        "update_memory",
        "delete_memory",
        # T14 team channel: directed delivery plus the recipient's inbox reader
        "deliver_memory",
        "check_inbox",
    ]


@pytest.mark.asyncio
async def test_update_memory_reports_note_without_content() -> None:
    class NoContentStore(CrudMemoryStore):
        def get(self, note_id: str) -> MemoryResponse:
            return MemoryResponse(success=True, memory_id=note_id, content=None)

    tool = UpdateMemoryTool(memory_store=NoContentStore())

    result = await tool.execute(memory_id="mem-1", content="New text.")

    assert result["success"] is False
    assert "no stored note" in result["error"]


# ---------------------------------------------------------------------------
# X-06: rewriting the body must not leave the replaced fact in `tags`
# ---------------------------------------------------------------------------


def _plate_note() -> MemoryNote:
    return MemoryNote(
        id="mem-plate",
        content="Y-C7-PLATE 我的车牌号是 京A8888，停车月卡登记用。",
        category="react_memory",
        keywords=["车牌号", "京A8888"],
        tags=["车牌号", "京A8888"],
    )


@pytest.mark.asyncio
async def test_update_memory_drops_tags_the_new_text_no_longer_supports() -> None:
    """X-06: 京A8888 stayed in tags after the body said 京A9999.

    The service stores whichever tag list a PUT carries (probed 2026-10-29),
    so the client has to stop shipping the replaced value.
    """
    store = CrudMemoryStore({"mem-plate": _plate_note()})
    tool = UpdateMemoryTool(memory_store=store)

    result = await tool.execute(
        memory_id="mem-plate",
        content="Y-C7-PLATE 我的车牌号是 京A9999，停车月卡登记用。",
    )

    assert result["success"] is True
    written = store.updated[0]
    assert written.tags == ["车牌号"]
    assert "京A8888" not in written.tags


@pytest.mark.asyncio
async def test_update_memory_clears_a_stale_keyword_fallback_too() -> None:
    """An empty tag list is refilled from keywords, so the keyword goes as well."""

    note = MemoryNote(
        id="mem-plate",
        content="我的车牌号是 京A8888。",
        category="react_memory",
        keywords=["京A8888"],
        tags=["京A8888"],
    )
    store = CrudMemoryStore({"mem-plate": note})
    tool = UpdateMemoryTool(memory_store=store)

    await tool.execute(memory_id="mem-plate", content="我的车牌号是 京A9999。")

    written = store.updated[0]
    assert written.tags == []
    assert "京A8888" not in written.keywords


@pytest.mark.asyncio
async def test_update_memory_keeps_tags_that_were_never_in_the_body() -> None:
    """A classification tag is not evidence, so a rewrite does not drop it."""
    note = MemoryNote(
        id="mem-1",
        content="Test memory 1",
        category="react_memory",
        keywords=["test", "sample"],
        tags=["important"],
    )
    store = CrudMemoryStore({"mem-1": note})
    tool = UpdateMemoryTool(memory_store=store)

    await tool.execute(memory_id="mem-1", content="Only content updated")

    assert store.updated[0].tags == ["important"]


@pytest.mark.asyncio
async def test_update_memory_keeps_the_old_tags_when_the_body_is_unchanged() -> None:
    store = CrudMemoryStore({"mem-plate": _plate_note()})
    tool = UpdateMemoryTool(memory_store=store)

    await tool.execute(memory_id="mem-plate", content=_plate_note().content)

    assert store.updated[0].tags == ["车牌号", "京A8888"]


def test_build_memory_tools_stamps_guidance_flag_on_context() -> None:
    class Ctx:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}

    context = Ctx()
    tools = build_memory_tools(
        memory_store=RecordingMemoryStore(), task="task", context=context
    )

    assert tools
    assert context.metadata["memory_tools_enabled"] is True

    disabled_context = Ctx()
    assert (
        build_memory_tools(memory_store=None, task="task", context=disabled_context)
        == []
    )
    assert "memory_tools_enabled" not in disabled_context.metadata


@pytest.mark.asyncio
async def test_search_memory_coerces_invalid_limit() -> None:
    store = RecordingMemoryStore()
    tool = SearchMemoryTool(memory_store=store)

    result = await tool.execute(query="anything", limit="lots")  # type: ignore[arg-type]

    assert result["success"] is True
    assert store.searches[0]["k"] == 5


@pytest.mark.asyncio
async def test_store_memory_falls_back_when_the_judged_write_is_unsupported() -> None:
    """A wrapper exposing ``remember`` over a local store must not fail the write."""

    class UnsupportedRememberStore(RecordingMemoryStore):
        def remember(self, note: MemoryNote) -> MemoryResponse:
            raise AttributeError("memory store does not support remember")

    store = UnsupportedRememberStore()
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="An insight.")

    assert result["success"] is True
    assert result["action"] == "create"
    assert len(store.added) == 1


class JudgedMemoryStore(RecordingMemoryStore):
    """Store exposing the judged ``remember`` write of the remote adapter.

    ``readable`` models what a follow-up GET can actually see: the service has
    answered merge/parent with ids that 404 (X-04, evidence
    ``y07-C9-merge.txt``), so a receipt id counts as stored only when this fake
    lists it.
    """

    def __init__(
        self,
        *,
        action: str,
        parent_id: str | None = None,
        success: bool = True,
        live: bool = True,
        degraded: Any = None,
        readable_creates: bool = True,
        receipt_body: str | None = None,
    ) -> None:
        super().__init__()
        self.action = action
        self.parent_id = parent_id
        self.success = success
        self.degraded = degraded
        self.readable_creates = readable_creates
        # A live id whose body is not the new text models the X-04 second
        # defect: existence alone is not evidence of a merge.
        self.receipt_body = receipt_body
        self.receipt_id = "mem-judged"
        self.readable: set[str] = {self.receipt_id} if live else set()
        self.notes: dict[str, MemoryNote] = {}
        self.remembered: list[MemoryNote] = []
        self.gets: list[str] = []

    def remember(self, note: MemoryNote) -> MemoryResponse:
        self.remembered.append(note)
        if not self.success:
            return MemoryResponse(success=False, error="judged write refused")
        metadata: dict[str, Any] = {
            "action": self.action,
            "parent_id": self.parent_id,
        }
        if self.degraded is not None:
            metadata["degraded"] = self.degraded
        # The judged write is what a real GET would return for this id.
        self.notes[self.receipt_id] = note
        return MemoryResponse(
            success=True,
            memory_id=self.receipt_id,
            metadata=metadata,
        )

    def get(self, note_id: str) -> MemoryResponse:
        self.gets.append(note_id)
        if note_id in self.readable:
            if note_id == self.receipt_id and self.receipt_body is not None:
                stored = _note(note_id, self.receipt_body)
            else:
                stored = self.notes.get(note_id) or _note(note_id, "stored")
            return MemoryResponse(
                success=True,
                memory_id=note_id,
                content=stored,
            )
        return MemoryResponse(
            success=False, error="Memory note not found or access denied"
        )

    def add(self, note: MemoryNote) -> MemoryResponse:
        self.added.append(note)
        if not self.add_success:
            return MemoryResponse(success=False, error="boom")
        memory_id = f"mem-{len(self.added)}"
        self.notes[memory_id] = note
        if self.readable_creates:
            self.readable.add(memory_id)
        return MemoryResponse(success=True, memory_id=memory_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "parent_id", "message_fragment"),
    [
        ("merge", None, "Merged into an existing similar memory"),
        ("parent", "mem-parent", "Attached as a child of a related memory"),
        ("create", None, "Stored as a new memory"),
    ],
)
async def test_store_memory_dispatches_the_judged_branch(
    action: str, parent_id: str | None, message_fragment: str
) -> None:
    """All three branches of the judged write are reported, none is dropped."""
    store = JudgedMemoryStore(action=action, parent_id=parent_id)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["stored"] is True
    assert result["action"] == action
    assert result["memory_id"] == "mem-judged"
    assert message_fragment in result["message"]
    assert store.remembered[0].content == "A durable business fact."
    assert store.remembered[0].metadata["source"] == "store_memory"
    if action == "parent":
        assert result["parent_id"] == "mem-parent"
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_reports_a_failed_judged_write() -> None:
    store = JudgedMemoryStore(action="create", success=False)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable fact.")

    assert result["success"] is False
    assert result["error"] == "judged write refused"
    assert tool.stored_count == 0
    assert store.added == []


@pytest.mark.asyncio
async def test_store_memory_emits_the_branch_in_the_store_trace() -> None:
    tracer = TraceEventRecorder()
    store = JudgedMemoryStore(action="parent", parent_id="mem-parent")
    tool = StoreMemoryTool(
        memory_store=store, task="task", runtime=FakeRuntime(tracer=tracer)
    )

    await tool.execute(content="A durable fact.")

    assert tracer.events[-1]["data"]["storage_action"] == "parent"
    assert tracer.events[-1]["data"]["storage_success"] is True
    assert tracer.events[-1]["data"]["storage_verified"] is True


# ---------------------------------------------------------------------------
# X-04: the judged receipt is verified before the model is told anything landed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["merge", "parent"])
async def test_store_memory_replaces_a_phantom_merge_receipt_with_a_create(
    action: str,
) -> None:
    """A receipt id that 404s must not be reported as a merge/attach.

    Evidence: y07-C9-merge.txt — action=merge returned e8bcc505/b46007db, both
    404, while the text lived under another id. The tool now re-stores the text
    as a new note and says so instead of promising a merge.
    """
    store = JudgedMemoryStore(action=action, parent_id="mem-parent", live=False)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["stored"] is True
    assert result["action"] == "create"
    assert result["memory_id"] == "mem-1"
    assert "Merged" not in result["message"]
    assert "Attached as a child" not in result["message"]
    assert "cannot be read back" in result["message"]
    assert "parent_id" not in result
    # The phantom receipt was checked, then the replacement was checked too.
    assert store.gets == ["mem-judged", "mem-1"]
    assert store.added[0].content == "A durable business fact."
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_reports_failure_when_the_create_retry_fails() -> None:
    store = JudgedMemoryStore(action="merge", live=False)
    store.add_success = False
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable fact.")

    assert result["success"] is False
    assert "stored" not in result
    assert "mem-judged" in result["error"]
    assert "Nothing was saved" in result["error"]
    assert tool.stored_count == 0


@pytest.mark.asyncio
async def test_store_memory_reports_a_replacement_that_also_disappears() -> None:
    store = JudgedMemoryStore(action="merge", live=False, readable_creates=False)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable fact.")

    assert result["success"] is False
    assert "replacement note mem-1 cannot be read back" in result["error"]
    assert tool.stored_count == 0


@pytest.mark.asyncio
async def test_store_memory_says_the_merge_is_unconfirmed_without_a_read_path() -> None:
    """A store that cannot read a note back keeps the branch but not the promise."""

    class NoReadStore(RecordingMemoryStore):
        def remember(self, note: MemoryNote) -> MemoryResponse:
            return MemoryResponse(
                success=True,
                memory_id="mem-remote",
                metadata={"action": "merge", "parent_id": None},
            )

    tool = StoreMemoryTool(memory_store=NoReadStore(), task="task")

    result = await tool.execute(content="A durable fact.")

    assert result["success"] is True
    assert result["action"] == "merge"
    assert "cannot read a note back" in result["message"]
    assert "Merged into an existing similar memory" not in result["message"]


@pytest.mark.asyncio
async def test_store_memory_reports_the_upstream_degraded_flag() -> None:
    store = JudgedMemoryStore(action="create", degraded=True)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable fact.")

    assert result["success"] is True
    assert result["degraded"] is True


@pytest.mark.asyncio
async def test_store_memory_omits_degraded_when_the_store_does_not_report_it() -> None:
    store = JudgedMemoryStore(action="create")
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable fact.")

    # A store with no such signal must not invent a "not degraded" claim.
    assert "degraded" not in result


class UnreadableReceiptStore(JudgedMemoryStore):
    """The read-back raises, which says nothing about whether the write landed."""

    def __init__(self, exc: BaseException) -> None:
        super().__init__(action="merge")
        self.exc = exc

    def get(self, note_id: str) -> MemoryResponse:
        self.gets.append(note_id)
        raise self.exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        # remote.py::get raises this for a 2.0s read timeout: no status_code.
        RemoteMemoryStoreError(
            "Remote memory upstream unreachable (request timed out after 2.0s "
            "-- a transport failure, not a credential rejection): GET "
            "/v1/memory/notes/mem-judged"
        ),
        # A 5xx is classified with the status attached by _raise_for_error.
        RemoteMemoryStoreError(
            "Remote memory service returned HTTP 503 on /v1/memory/notes/mem-judged",
            status_code=503,
        ),
        # A connection reset also arrives as a plain transport error.
        RemoteMemoryStoreError(
            "Remote memory upstream unreachable (connection failure -- not a "
            "credential rejection): GET /v1/memory/notes/mem-judged"
        ),
    ],
)
async def test_store_memory_does_not_rewrite_when_the_receipt_is_unreadable(
    exc: RemoteMemoryStoreError,
) -> None:
    """X-04: a timeout/5xx read says nothing about the write — never write twice.

    The judged write was accepted; a failed GET must not be read as "the note
    is gone". Re-creating it wrote the same fact twice (X-04 rework), so the
    tool reports accepted-but-unverified and leaves the store alone.
    """
    tracer = TraceEventRecorder()
    store = UnreadableReceiptStore(exc)
    tool = StoreMemoryTool(
        memory_store=store, task="task", runtime=FakeRuntime(tracer=tracer)
    )

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["stored"] is True
    assert result["verified"] is False
    assert result["action"] == "merge"
    assert result["message"] == UNVERIFIED_RECEIPT_MESSAGE
    assert "Merged" not in result["message"]
    assert store.added == []
    assert store.gets == ["mem-judged"]
    assert tracer.events[-1]["data"]["storage_verified"] is False
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_retries_a_create_when_the_receipt_read_raises_404() -> None:
    """A raised 404 is positive evidence of absence, so the create retry stands."""

    class Raising404Store(JudgedMemoryStore):
        def get(self, note_id: str) -> MemoryResponse:
            self.gets.append(note_id)
            if note_id == self.receipt_id:
                raise RemoteMemoryStoreError(
                    "Remote memory service returned HTTP 404（code=not_found） on "
                    f"/v1/memory/notes/{note_id}",
                    status_code=404,
                )
            stored = self.notes.get(note_id) or _note(note_id, "stored")
            return MemoryResponse(success=True, memory_id=note_id, content=stored)

    store = Raising404Store(action="merge")
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["action"] == "create"
    assert result["memory_id"] == "mem-1"
    assert len(store.added) == 1
    assert "Merged" not in result["message"]
    assert store.gets == ["mem-judged", "mem-1"]


@pytest.mark.asyncio
async def test_store_memory_recreates_when_the_receipt_note_lacks_the_new_text() -> None:
    """X-04: a live id is not proof of a merge — the body has to carry the text."""
    store = JudgedMemoryStore(
        action="merge", receipt_body="an unrelated older fact"
    )
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["stored"] is True
    assert result["action"] == "create"
    assert result["memory_id"] == "mem-1"
    assert result["verified"] is True
    assert "Merged" not in result["message"]
    assert "does not contain the new text" in result["message"]
    assert len(store.added) == 1
    assert store.added[0].content == "A durable business fact."
    assert store.gets == ["mem-judged", "mem-1"]


@pytest.mark.asyncio
async def test_store_memory_reports_both_errors_when_the_recreate_fails() -> None:
    store = JudgedMemoryStore(action="merge", receipt_body="an older fact")
    store.add_success = False
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable fact.")

    assert result["success"] is False
    assert "does not contain the new text" in result["error"]
    assert "boom" in result["error"]
    assert "Nothing was saved" in result["error"]
    assert tool.stored_count == 0


@pytest.mark.asyncio
async def test_store_memory_still_reports_a_merge_when_the_body_carries_it() -> None:
    """The fix must not turn a legitimate merge into a create.

    The fake's GET returns the note the judged write claimed to have merged
    the text into, so the body contains the new text and "Merged" stands.
    """
    store = JudgedMemoryStore(action="merge")
    store.receipt_body = "earlier note\nA durable business fact."
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["action"] == "merge"
    assert result["verified"] is True
    assert "Merged into an existing similar memory" in result["message"]
    assert store.added == []


@pytest.mark.asyncio
async def test_store_memory_normalises_whitespace_when_checking_the_body() -> None:
    """A body that differs only in line breaks still counts as carrying the text."""
    store = JudgedMemoryStore(action="merge")
    store.receipt_body = "earlier note\nA durable\n  business fact."
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["action"] == "merge"
    assert result["verified"] is True
    assert "Merged" in result["message"]


class NoIdJudgedStore(JudgedMemoryStore):
    """A receipt that answers success but carries no memory id."""

    def __init__(self, action: str = "merge") -> None:
        super().__init__(action=action)

    def remember(self, note: MemoryNote) -> MemoryResponse:
        self.remembered.append(note)
        return MemoryResponse(
            success=True,
            memory_id=None,
            metadata={"action": self.action, "parent_id": None},
        )

    def get(self, note_id: str) -> MemoryResponse:  # pragma: no cover - asserted
        self.gets.append(note_id)
        return MemoryResponse(
            success=True, memory_id=note_id, content=_note(note_id, "x")
        )


@pytest.mark.asyncio
async def test_store_memory_treats_a_receipt_without_id_as_unconfirmed() -> None:
    """No id is not a not-found answer: do not create a second copy."""
    store = NoIdJudgedStore()
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["stored"] is True
    assert result["verified"] is False
    assert "回执未带 memory id" in result["message"]
    # The timeout/5xx wording would claim a read-back that never happened.
    assert "写入已受理但未能确认落库" not in result["message"]
    assert store.gets == []
    assert store.added == []
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_does_not_rewrite_when_a_401_mentions_not_found() -> None:
    """A 401 whose upstream message embeds "not found" is not an absence verdict.

    The read-back was refused, so the message must say exactly that: the whole
    text is asserted, including the parenthesis, because the previous wording
    enumerated "timed out / failed to connect / 5xx" and the model copied the
    wrong cause back (third review).
    """
    exc = RemoteMemoryStoreError(
        "上游返回 HTTP 401（code=unauthorized） on /v1/memory/notes/mem-judged，"
        "上游 message: identity not found or access denied",
        status_code=401,
    )
    store = UnreadableReceiptStore(exc)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["verified"] is False
    assert result["action"] == "merge"
    assert result["message"] == READBACK_REJECTED_MESSAGE
    for forbidden in ("timed out", "timeout", "5xx", "超时"):
        assert forbidden not in result["message"]
    assert "Merged" not in result["message"]
    assert store.added == []
    assert store.gets == ["mem-judged"]


@pytest.mark.asyncio
async def test_store_memory_does_not_rewrite_when_the_readback_is_a_401() -> None:
    """A plain credential rejection on the read-back is not a timeout.

    ``remote.py::get`` raises this exact class for a 401; the write may have
    landed, so the tool still refuses to write again, but the wording must name
    the rejection rather than a transport failure.
    """
    exc = RemoteMemoryStoreAuthenticationError(
        "上游拒绝了服务凭据：HTTP 401（code=unauthorized） on "
        "/v1/memory/notes/mem-judged。上游明示凭据类错误，请核对 "
        "XAGENT_MEMORY_SERVICE_KEY（无效、已吊销或不属于该实例）。",
        status_code=401,
    )
    store = UnreadableReceiptStore(exc)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["verified"] is False
    assert result["action"] == "merge"
    assert result["message"] == READBACK_REJECTED_MESSAGE
    for forbidden in ("timed out", "timeout", "5xx", "超时"):
        assert forbidden not in result["message"]
    assert store.added == []
    assert store.gets == ["mem-judged"]


@pytest.mark.asyncio
async def test_store_memory_does_not_rewrite_when_the_2xx_body_is_unparsable() -> None:
    """A 2xx read-back whose body is not JSON is unreadable, not a timeout.

    ``remote.py::_json`` raises this message with no status attached, so the
    class has to come from the text; the write may still have landed.
    """
    exc = RemoteMemoryStoreError(
        "Remote memory service returned a non-JSON response: "
        "/v1/memory/notes/mem-judged"
    )
    store = UnreadableReceiptStore(exc)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["verified"] is False
    assert result["action"] == "merge"
    assert result["message"] == READBACK_REJECTED_MESSAGE
    for forbidden in ("timed out", "timeout", "5xx", "超时"):
        assert forbidden not in result["message"]
    assert store.added == []
    assert store.gets == ["mem-judged"]


@pytest.mark.asyncio
async def test_store_memory_claims_no_cause_when_the_readback_cannot_be_classified() -> None:
    """An unclassifiable read-back states no cause instead of guessing one.

    The message still stops a duplicate write, but it must not name a timeout,
    a connection failure or a 5xx it has no evidence for.
    """
    store = UnreadableReceiptStore(RuntimeError("boom"))
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["verified"] is False
    assert result["message"] == RECEIPT_FAILURE_UNKNOWN_MESSAGE
    for forbidden in ("timed out", "timeout", "5xx", "超时"):
        assert forbidden not in result["message"]
    assert store.added == []
    assert store.gets == ["mem-judged"]


@pytest.mark.asyncio
async def test_store_memory_reports_an_unconfirmed_replacement_body() -> None:
    """X-04 rework: a created replacement whose body differs is not a timeout.

    The create answered and the note is read back, so the timeout/5xx wording
    (accepted but unverified) would misdescribe the state. The replacement
    exists and must not be written again.
    """
    store = JudgedMemoryStore(action="merge", receipt_body="an unrelated older fact")

    def get(note_id: str) -> MemoryResponse:
        store.gets.append(note_id)
        body = (
            store.receipt_body
            if note_id == store.receipt_id
            else "a different stored fact"
        )
        return MemoryResponse(
            success=True, memory_id=note_id, content=_note(note_id, body or "")
        )

    store.get = get  # type: ignore[method-assign]
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["action"] == "create"
    assert result["verified"] is False
    assert "已新建记忆但未能确认正文一致" in result["message"]
    assert "写入已受理但未能确认落库" not in result["message"]
    assert "Merged" not in result["message"]
    assert len(store.added) == 1
    assert store.gets == ["mem-judged", "mem-1"]
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_accepts_a_create_receipt_case_and_spacing() -> None:
    """A create receipt whose body differs only in case/spacing is verified."""
    store = JudgedMemoryStore(action="create")
    store.receipt_body = "EARLIER NOTE\na   durable\nbusiness FACT."
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["action"] == "create"
    assert result["verified"] is True
    assert result["message"].startswith("Stored as a new memory")
    assert store.added == []


@pytest.mark.asyncio
async def test_store_memory_never_recreates_a_create_receipt_body() -> None:
    """X-04 rework: a create receipt's body mismatch never triggers a second create.

    The note it names exists; only the body comparison failed. Writing again
    would duplicate the same fact, so the write is reported unconfirmed.
    """
    store = JudgedMemoryStore(action="create", receipt_body="an unrelated older fact")
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable business fact.")

    assert result["success"] is True
    assert result["action"] == "create"
    assert result["verified"] is False
    assert "已写入但未能确认正文一致" in result["message"]
    assert "写入已受理但未能确认落库" not in result["message"]
    assert store.added == []
    assert store.gets == ["mem-judged"]


# ---------------------------------------------------------------------------
# X-06: merging locally must not keep tags the superseded text supported
# ---------------------------------------------------------------------------


class LocalMergeStore(RecordingMemoryStore):
    """Local store shape: dedup search finds a note, update rewrites it."""

    def __init__(self, existing: MemoryNote) -> None:
        super().__init__(search_results=[existing])
        self.updated: list[MemoryNote] = []

    def update(self, note: MemoryNote) -> MemoryResponse:
        self.updated.append(note)
        return MemoryResponse(success=True, memory_id=note.id)


@pytest.mark.asyncio
async def test_merge_existing_keeps_tags_the_merged_body_still_supports() -> None:
    """X-04 rework: a merge appends to the old body, so old tags stay supported.

    Charging ``drop_replaced_tags`` with the incoming fragment alone cleared a
    value the merged note still cites in its retained history (the plate
    number below). The merge path clears nothing it cannot prove replaced.
    """
    existing = MemoryNote(
        id="mem-1",
        content="Y-C7 我的车牌号是 京A8888，停车月卡登记用。",
        keywords=["车牌号", "京A8888"],
        tags=["车牌号", "京A8888"],
    )
    store = LocalMergeStore(existing)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="Y-C7 我的车牌号已换成 京A9999。")

    assert result["success"] is True
    assert result["action"] == "update"
    written = store.updated[0]
    # The old statement is kept as history, so both values are in the body.
    assert "京A9999" in str(written.content)
    assert "京A8888" in str(written.content)
    assert written.tags == ["车牌号", "京A8888"]
    assert written.keywords == ["车牌号", "京A8888"]


@pytest.mark.asyncio
async def test_merge_existing_does_not_clear_a_classification_tag() -> None:
    """A classification tag the old body supported survives the merge."""
    existing = MemoryNote(
        id="mem-1",
        content="部署脚本在 scripts/deploy.sh，execution 分类使用。",
        keywords=["deploy"],
        tags=["execution", "deploy"],
    )
    store = LocalMergeStore(existing)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="今天改用 make release 发布。")

    assert result["success"] is True
    written = store.updated[0]
    assert written.tags == ["execution", "deploy"]
    assert written.keywords == ["deploy"]


# ---------------------------------------------------------------------------
# X-05: a 409 judged write degrades to a plain create instead of vanishing
# ---------------------------------------------------------------------------

APPEND_ONLY_409 = (
    "Remote memory service returned HTTP 409（code=conflict） on "
    "/v1/memory/remember，上游 message: delivered notes are append-only: "
    "send a follow-up delivery instead"
)


class AppendOnlyRememberStore(RecordingMemoryStore):
    """Judged write refused with the service's append-only 409."""

    def __init__(self, *, exc: BaseException, add_success: bool = True) -> None:
        super().__init__(add_success=add_success)
        self.exc = exc
        self.remembered: list[MemoryNote] = []

    def remember(self, note: MemoryNote, *, force: bool = False) -> MemoryResponse:
        self.remembered.append(note)
        raise self.exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        RemoteMemoryStoreError(APPEND_ONLY_409, status_code=409),
        RuntimeError(APPEND_ONLY_409),
    ],
    ids=["status-code", "message-text"],
)
async def test_store_memory_retries_an_append_only_conflict_as_a_create(
    exc: BaseException,
) -> None:
    """za03-case2: the preference write was refused and silently lost."""
    store = AppendOnlyRememberStore(exc=exc)
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(
        content="团队规定：报销/出差申请必须提前三天提交申请。",
        kind="domain_insight",
    )

    assert result["success"] is True
    assert result["stored"] is True
    assert result["action"] == "create"
    assert result["memory_id"] == "mem-1"
    assert "Merged" not in result["message"]
    # The retry carries no merge target: it is a plain create of the same text.
    assert store.added[0].content == "团队规定：报销/出差申请必须提前三天提交申请。"
    assert store.added[0].metadata == {
        "kind": "domain_insight",
        "source": "store_memory",
    }
    assert tool.stored_count == 1


@pytest.mark.asyncio
async def test_store_memory_surfaces_the_conflict_when_the_create_retry_fails() -> None:
    store = AppendOnlyRememberStore(
        exc=RemoteMemoryStoreError(APPEND_ONLY_409, status_code=409),
        add_success=False,
    )
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable fact.")

    assert result["success"] is False
    assert "append-only" in result["error"]
    assert "retried as a new memory without a merge target" in result["error"]
    assert "boom" in result["error"]
    assert tool.stored_count == 0


@pytest.mark.asyncio
async def test_store_memory_does_not_retry_other_write_failures() -> None:
    """Only the append-only conflict has a different form that can succeed."""

    class RateLimitedStore(RecordingMemoryStore):
        def remember(self, note: MemoryNote) -> MemoryResponse:
            raise RuntimeError("Remote memory service returned HTTP 429 — slow down")

    store = RateLimitedStore()
    tool = StoreMemoryTool(memory_store=store, task="task")

    result = await tool.execute(content="A durable fact.")

    assert result["success"] is False
    assert "429" in result["error"]
    assert store.added == []


# ---------------------------------------------------------------------------
# check_inbox: the dedicated inbox surface behind the wrapper chain (P0-1)
# ---------------------------------------------------------------------------


def _delivery(note_id: str, content: str, at: str, from_user: int) -> MemoryNote:
    return MemoryNote(
        id=note_id,
        content=content,
        category="general",
        timestamp=datetime.fromisoformat(at),
        metadata={"deliver": True, "from_user": from_user},
    )


def _inbox_notes() -> list[MemoryNote]:
    return [
        _delivery("d1", "first", "2026-10-15T09:00:00", 7),
        _delivery("d2", "second", "2026-10-15T10:00:00", 8),
        _delivery("d3", "third", "2026-10-15T11:00:00", 9),
    ]


class InboxSurfaceStore:
    """Fake exposing the dedicated inbox surface (unread list + receipts)."""

    def __init__(self) -> None:
        self.notes = _inbox_notes()
        self.inbox_calls: list[bool] = []
        self.read_ids: list[str] = []
        self.list_all_calls: list[Any] = []

    def inbox_list(self, include_read: bool = False) -> list[MemoryNote]:
        self.inbox_calls.append(include_read)
        return list(self.notes)

    def inbox_read(self, ids: list[str]) -> int:
        self.read_ids = list(ids)
        return len(ids)

    def list_all(self, filters: Any = None) -> list[MemoryNote]:
        self.list_all_calls.append(filters)
        raise AssertionError("legacy list_all must not run when the surface exists")


class LegacyInboxStore:
    """Fake without the dedicated surface: only the legacy list path exists."""

    def __init__(self) -> None:
        self.notes = _inbox_notes()
        self.list_all_calls: list[Any] = []

    def list_all(self, filters: Any = None) -> list[MemoryNote]:
        self.list_all_calls.append(filters)
        return list(self.notes)


class _ScriptedSession:
    """Fake HTTP transport: scripted ``(status, payload)`` answers in order."""

    def __init__(self, *responses: tuple[int, Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        status, payload = self._responses.pop(0)
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode("utf-8")
        response.headers["Content-Type"] = "application/json"
        return response


class _WrappedRemoteStore:
    """A wrapper that does not forward the inbox surface, adapter behind it."""

    def __init__(self, remote: RemoteMemoryStore) -> None:
        self._base_store = remote
        self.list_all_calls: list[Any] = []
        self.added: list[MemoryNote] = []

    def list_all(self, filters: Any = None) -> list[MemoryNote]:
        self.list_all_calls.append(filters)
        return [_delivery("d1", "legacy", "2026-10-15T09:00:00", 7)]

    def add(self, note: MemoryNote) -> MemoryResponse:
        self.added.append(note)
        raise RemoteMemoryStoreError(
            "Remote memory service returned HTTP 409: /v1/memory/deliver"
            " — you already delivered this note;"
            " send a follow-up delivery instead",
            status_code=409,
        )


class _PublicationSource:
    """Minimal publication source for the revocable wrapper (test double)."""

    def __init__(self, generation: int = 1) -> None:
        self.generation = generation

    def publication_generation(self) -> int:
        return self.generation

    def published_status(self) -> MemoryLifecycleStatus:
        return MemoryLifecycleStatus(state=MemoryLifecycleState.RESTART_REQUIRED)


@pytest.mark.asyncio
async def test_check_inbox_walks_the_wrapper_to_the_dedicated_surface() -> None:
    """A wrapper store must not hide the surface: unread filter + receipts."""
    base = InboxSurfaceStore()
    tool = CheckInboxTool(memory_store=UserIsolatedMemoryStore(base))

    result = await tool.execute(limit=2)

    assert result["success"] is True
    assert [d["id"] for d in result["deliveries"]] == ["d3", "d2"]
    assert result["count"] == 2
    assert result["marked_read"] == 2
    assert result["unread_remaining"] == 1
    assert base.read_ids == ["d3", "d2"]
    assert base.inbox_calls == [False]
    # the legacy path must not be taken when the surface exists
    assert base.list_all_calls == []


@pytest.fixture
def memory_identity() -> Any:
    """R6: remote memory I/O requires a trusted identity header."""
    token = current_user_id.set(7)
    yield
    current_user_id.reset(token)


@pytest.mark.asyncio
async def test_check_inbox_through_the_full_production_chain(memory_identity: Any) -> None:
    """Revocable(user_isolated(remote)): live reads go over the inbox routes,
    and a revoked reference fails closed instead of reaching the adapter."""
    session = _ScriptedSession(
        (
            200,
            [
                {
                    "id": "d1",
                    "content": "hello",
                    "category": "general",
                    "timestamp": "2026-10-15T10:00:00",
                    "metadata": {"from_user": 7},
                }
            ],
        ),
        (200, {"marked": 1}),
    )
    remote = RemoteMemoryStore(SERVICE_URL, SERVICE_KEY, session=session)
    source = _PublicationSource(generation=1)
    store = RevocableMemoryStore(  # type: ignore[arg-type]
        UserIsolatedMemoryStore(remote), source, generation=1
    )
    tool = CheckInboxTool(memory_store=store)

    live = await tool.execute()

    assert live["success"] is True
    assert [d["id"] for d in live["deliveries"]] == ["d1"]
    assert live["marked_read"] == 1
    assert live["unread_remaining"] == 0
    assert [call["url"] for call in session.calls] == [
        f"{SERVICE_URL}/v1/memory/inbox/list",
        f"{SERVICE_URL}/v1/memory/inbox/read",
    ]

    source.generation = 2  # the manager revoked this publication

    revoked = await tool.execute()

    assert revoked["success"] is False
    # the revoked reference never reached the adapter: no third request
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_check_inbox_reaches_the_adapter_behind_a_hiding_wrapper(memory_identity: Any) -> None:
    """A wrapper without forwarding methods must not hide the capability."""
    session = _ScriptedSession(
        (
            200,
            [
                {
                    "id": "d1",
                    "content": "hello",
                    "category": "general",
                    "timestamp": "2026-10-15T10:00:00",
                    "metadata": {"from_user": 7},
                }
            ],
        ),
        (200, {"marked": 1}),
    )
    wrapper = _WrappedRemoteStore(
        RemoteMemoryStore(SERVICE_URL, SERVICE_KEY, session=session)
    )
    tool = CheckInboxTool(memory_store=wrapper)

    result = await tool.execute()

    assert result["success"] is True
    assert [d["id"] for d in result["deliveries"]] == ["d1"]
    assert result["marked_read"] == 1
    assert result["unread_remaining"] == 0
    assert wrapper.list_all_calls == []
    assert [call["url"] for call in session.calls] == [
        f"{SERVICE_URL}/v1/memory/inbox/list",
        f"{SERVICE_URL}/v1/memory/inbox/read",
    ]


@pytest.mark.asyncio
async def test_check_inbox_falls_back_to_legacy_on_a_missing_inbox_route(memory_identity: Any) -> None:
    """An old service answers 404 for the inbox routes: use the legacy path."""
    session = _ScriptedSession((404, {"error": {"message": "no such route"}}))
    wrapper = _WrappedRemoteStore(
        RemoteMemoryStore(SERVICE_URL, SERVICE_KEY, session=session)
    )
    tool = CheckInboxTool(memory_store=wrapper)

    result = await tool.execute()

    assert result["success"] is True
    assert [d["id"] for d in result["deliveries"]] == ["d1"]
    # legacy shape: no receipt bookkeeping, no unread counter
    assert "unread_remaining" not in result
    assert "marked_read" not in result
    assert wrapper.list_all_calls == [{"metadata": {"deliver": True}}]
    assert session.calls[0]["url"] == f"{SERVICE_URL}/v1/memory/inbox/list"


@pytest.mark.asyncio
async def test_check_inbox_falls_back_to_legacy_when_the_delegate_lacks_it() -> None:
    """A forwarding wrapper over a local store raises AttributeError → legacy."""
    base = LegacyInboxStore()
    tool = CheckInboxTool(memory_store=UserIsolatedMemoryStore(base))

    result = await tool.execute()

    assert result["success"] is True
    assert result["count"] == 3
    assert "unread_remaining" not in result
    assert base.list_all_calls == [{"metadata": {"deliver": True}}]


@pytest.mark.asyncio
async def test_check_inbox_keeps_the_legacy_path_for_surface_less_fakes() -> None:
    """A plain fake store with only list_all keeps the historical behaviour."""
    base = LegacyInboxStore()
    tool = CheckInboxTool(memory_store=base)

    result = await tool.execute(limit=2)

    assert result == {
        "success": True,
        "count": 2,
        "deliveries": [
            {
                "id": "d3",
                "content": "third",
                "from_user": 9,
                "timestamp": "2026-10-15 11:00:00",
            },
            {
                "id": "d2",
                "content": "second",
                "from_user": 8,
                "timestamp": "2026-10-15 10:00:00",
            },
        ],
    }
    assert base.list_all_calls == [{"metadata": {"deliver": True}}]


# ---------------------------------------------------------------------------
# server-side error messages reach the model (P1-4)
# ---------------------------------------------------------------------------

APPEND_ONLY_409 = (
    "Remote memory service returned HTTP 409: /v1/memory/notes/mem-1"
    " — this note is append-only; send a follow-up delivery instead"
)


@pytest.mark.asyncio
async def test_update_memory_surfaces_the_service_message() -> None:
    class AppendOnlyStore(CrudMemoryStore):
        def update(self, note: MemoryNote) -> MemoryResponse:
            raise RemoteMemoryStoreError(APPEND_ONLY_409, status_code=409)

    store = AppendOnlyStore({"mem-1": _note("mem-1", "Delivered note.")})
    tool = UpdateMemoryTool(memory_store=store)

    result = await tool.execute(memory_id="mem-1", content="Corrected fact.")

    assert result["success"] is False
    assert "send a follow-up delivery instead" in result["error"]


@pytest.mark.asyncio
async def test_delete_memory_surfaces_the_service_message() -> None:
    class ConflictingDeleteStore(CrudMemoryStore):
        def delete(self, note_id: str) -> MemoryResponse:
            raise RemoteMemoryStoreError(APPEND_ONLY_409, status_code=409)

    tool = DeleteMemoryTool(memory_store=ConflictingDeleteStore({"mem-1": _note("mem-1", "x")}))

    result = await tool.execute(memory_id="mem-1")

    assert result["success"] is False
    assert "send a follow-up delivery instead" in result["error"]


@pytest.mark.asyncio
async def test_store_memory_surfaces_the_service_message_on_judged_write() -> None:
    """A refused judged write that has no repair path keeps the service's text."""

    class RejectedRememberStore(RecordingMemoryStore):
        def remember(self, note: MemoryNote, *, force: bool = False) -> MemoryResponse:
            raise RemoteMemoryStoreError(
                "Remote memory service returned HTTP 422（code=invalid） on "
                "/v1/memory/remember，上游 message: content rejected",
                status_code=422,
            )

    tool = StoreMemoryTool(memory_store=RejectedRememberStore(), task="task")

    result = await tool.execute(content="A durable fact.")

    assert result["success"] is False
    assert "content rejected" in result["error"]


@pytest.mark.asyncio
async def test_deliver_memory_surfaces_the_service_message() -> None:
    wrapper = _WrappedRemoteStore(RemoteMemoryStore(SERVICE_URL, SERVICE_KEY))
    tool = DeliverMemoryTool(memory_store=wrapper)

    result = await tool.execute(target="t10-probe", content="Ship it.")

    assert result["success"] is False
    assert "send a follow-up delivery instead" in result["error"]


@pytest.mark.asyncio
async def test_search_memory_surfaces_the_service_message() -> None:
    class RateLimitedStore(CrudMemoryStore):
        def search(self, **kwargs: Any) -> list[Any]:
            raise RemoteMemoryStoreError(
                "Remote memory service returned HTTP 429: /v1/memory/search"
                " — rate limited, slow down",
                status_code=429,
            )

    tool = SearchMemoryTool(memory_store=RateLimitedStore())

    result = await tool.execute(query="anything")

    assert result["success"] is False
    assert "rate limited, slow down" in result["error"]


# ---------------------------------------------------------------------------
# search_memory space / category filters (P1-6a, client half)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_memory_forwards_space_and_category_filters() -> None:
    store = CrudMemoryStore()
    tool = SearchMemoryTool(memory_store=store)

    filtered = await tool.execute(
        query="budget", space="department", category="react_memory"
    )
    unfiltered = await tool.execute(query="budget")

    assert filtered["success"] is True
    assert unfiltered["success"] is True
    assert store.searches[0]["filters"] == {
        "space": "department",
        "category": "react_memory",
    }
    # default behaviour is unchanged: no caller filters at all
    assert store.searches[1]["filters"] is None


@pytest.mark.asyncio
async def test_search_memory_rejects_an_unknown_space() -> None:
    store = CrudMemoryStore()
    tool = SearchMemoryTool(memory_store=store)

    result = await tool.execute(query="budget", space="team")

    assert result["success"] is False
    assert "personal" in result["error"] and "department" in result["error"]
    assert store.searches == []


def test_search_memory_schema_advertises_space_and_category() -> None:
    properties = SearchMemoryArgs.model_json_schema()["properties"]
    space_schema = json.dumps(properties["space"])
    for value in ("personal", "department", "company"):
        assert value in space_schema
    assert "category" in properties
