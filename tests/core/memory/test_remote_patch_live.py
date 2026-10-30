"""Optional actual-memsrv/pgvector contract; uses only the synthetic Go bridge.

Start tests/fixtures/memsrv_patch_bridge_test.go in its audited server checkout,
then set XAGENT_CONTRACT_BRIDGE_DIR to the shared temporary bridge directory.
This suite never creates users/keys or contacts a non-loopback server.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

from xagent.core.execution_scope import (
    ExecutionScope,
    reset_execution_scope,
    set_execution_scope,
)
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.remote import RemoteMemoryStore, RemoteMemoryStoreError


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1", "localhost", "::1"])
def test_actual_memsrv_nonstructural_patch_contract():
    bridge_value = os.getenv("XAGENT_CONTRACT_BRIDGE_DIR")
    if not bridge_value:
        pytest.skip("isolated memsrv bridge not configured")
    bridge = Path(bridge_value)
    url = (bridge / "url").read_text().strip()
    assert urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}
    headers = {"Authorization": "Bearer fixture-service", "X-User-External-Id": "31"}

    class RecordingSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.trust_env = False
            self.calls = []

        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs.get("json")))
            return super().request(method, url, **kwargs)

    session = RecordingSession()
    control = requests.Session()
    control.trust_env = False

    def raw(note_id):
        result = control.get(
            f"{url}/v1/memory/notes/{note_id}", headers=headers, timeout=3
        )
        result.raise_for_status()
        return result.json()

    ids = ["patch-root", "patch-child", "patch-grand"]
    before = {note_id: raw(note_id) for note_id in ids}
    assert all(
        note["metadata"].get("xagent_contract_fixture") is True
        for note in before.values()
    )
    structure = {
        note_id: {
            key: before[note_id].get(key)
            for key in (
                "parent_id",
                "path",
                "owner_id",
                "title",
                "kind",
                "see_also",
                "supersedes",
                "created_at",
            )
        }
        for note_id in ids
    }
    store = RemoteMemoryStore(
        url, "fixture-service", session=session, user_id_provider=lambda: 31
    )
    token = set_execution_scope(
        ExecutionScope(memory_dimensions={"client_application_id": "42"})
    )
    try:
        assert (
            control.get(url + "/v1/org/users", headers=headers, timeout=3).status_code
            == 403
        )
        for note_id in ids:
            note = store.get(note_id).content
            assert isinstance(note, MemoryNote)
            note.content = f"edited {note_id}"
            note.tags = ["edited"]
            note.keywords = []
            note.metadata["business"] = "updated"
            ack = store.update(note)
            assert ack.success and ack.content is None
            assert isinstance(ack.metadata["degraded"], bool)
            got = raw(note_id)
            assert got["tags"] == ["edited"] and got["keywords"] == []
            assert got["metadata"]["user_id"] == "31"
            assert got["metadata"]["execution_scope_client_application_id"] == "42"
            assert {key: got.get(key) for key in structure[note_id]} == structure[
                note_id
            ]
        for binary in [b"\x00\xff", b""]:
            note = store.get("patch-child").content
            note.content = binary
            note.mime_type = "application/octet-stream"
            assert store.update(note).success
            assert store.get("patch-child").content.content == binary
        note = store.get("patch-root").content
        note.metadata = {}
        assert store.update(note).success
        assert raw("patch-root")["metadata"]["execution_scope_agent"] == 10
        assert "business" not in raw("patch-root")["metadata"]
        note = store.get("patch-root").content
        note.metadata["execution_scope_agent"] = 11
        with pytest.raises(RemoteMemoryStoreError) as failed:
            store.update(note)
        assert failed.value.status_code == 400
        assert raw("patch-root")["metadata"]["execution_scope_agent"] == 10
        denied = RemoteMemoryStore(
            url, "fixture-service", session=session, user_id_provider=lambda: 33
        )
        assert not denied.update(
            MemoryNote(id="patch-root", content="must not write")
        ).success
        # Retain the note in the client before a separate writer reparents it.
        stale = store.get("patch-child").content
        (bridge / "ordinary-done").write_text("ready")
        deadline = time.monotonic() + 30
        while not (bridge / "moved").exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        moved = {
            note_id: {key: raw(note_id).get(key) for key in ("parent_id", "path")}
            for note_id in ids
        }
        assert moved["patch-child"]["parent_id"] == "patch-extra"
        stale.content = "body edit after another writer reparented"
        assert store.update(stale).success
        assert {
            note_id: {key: raw(note_id).get(key) for key in ("parent_id", "path")}
            for note_id in ids
        } == moved
        assert all(method != "PUT" for method, _url, _body in session.calls)
        for method, _url, body in session.calls:
            if method == "PATCH":
                assert not (
                    {"id", "parent_id", "path", "space_kind", "space_id", "owner_id"}
                    & body.keys()
                )
        (bridge / "done").write_text("passed")
    finally:
        reset_execution_scope(token)
        session.close()
        control.close()
