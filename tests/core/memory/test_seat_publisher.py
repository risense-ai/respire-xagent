"""Real local publisher/reader IPC; source verification is synthetic and explicit."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from xagent.core.memory import agent_seats
from xagent.core.memory.agent_seats import AgentSeatRequest
from xagent.core.memory.remote import (
    RemoteMemoryStore,
    RemoteMemoryStoreIdentityError,
    delegated_agent_identity,
)
from xagent.core.memory.seat_file_provider import FileAgentSeatProvider
from xagent.core.memory.seat_publisher import TrustedSeatPublisher, VerifiedSeatGrant
from xagent.core.user_context import current_user_id

URL, KEY = "http://synthetic-memsrv", "synthetic-runtime-only"
REQUEST = AgentSeatRequest(URL, "31", hashlib.sha256(KEY.encode()).hexdigest())
GRANT = VerifiedSeatGrant(REQUEST, frozenset({"7"}), "a" * 64)


@pytest.fixture
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    path, journal = tmp_path / "grants.json", tmp_path / "journal.json"
    source = Mock(return_value=(GRANT,))
    publisher = TrustedSeatPublisher(path, journal, source)
    publisher.start()
    reader = FileAgentSeatProvider(path)
    assert reader.refresh()
    monkeypatch.setattr(agent_seats, "_provider", reader)
    monkeypatch.setattr(agent_seats, "_provider_generation", 1)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", URL)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", KEY)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "true")
    transport = Mock(spec=requests.Session)
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"results": []}'
    transport.request.return_value = response
    store = RemoteMemoryStore(URL, KEY, session=transport)
    token = current_user_id.set(31)
    try:
        yield path, journal, source, publisher, reader, store, transport
    finally:
        current_user_id.reset(token)
        reader.close()
        publisher.close()


def test_renewal_and_unrelated_grant_do_not_end_long_scope(
    live: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _journal, source, publisher, reader, store, transport = live
    initial = reader(REQUEST)
    clock = [time.time(), time.monotonic()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "monotonic", lambda: clock[1])
    with delegated_agent_identity(7, store=store):
        for step in range(120):
            clock[0] += 1
            clock[1] += 1
            if step % 10 == 0:
                assert publisher.refresh()
            assert reader.refresh()
            store.search("synthetic long task")
        assert clock[1] > initial.valid_until
        source.return_value = (
            GRANT,
            replace(GRANT, request=replace(REQUEST, user_external_id="32")),
        )
        assert publisher.refresh() and reader.refresh()
        store.search("another user's grant changed")
        assert reader(REQUEST).generation == initial.generation
        assert reader(REQUEST).valid_until > initial.valid_until
        assert json.loads(path.read_bytes())["revision"] > 10
    assert transport.request.call_count == 121


@pytest.mark.parametrize(
    "change",
    [
        "authorization",
        "agent_set",
        "revoke",
        "source_error",
        "bad_grant",
        "lease_gap",
        "reader_gap",
    ],
)
def test_recovery_never_revives_old_scope(live: Any, change: str) -> None:
    _path, _journal, source, publisher, reader, store, transport = live
    with delegated_agent_identity(7, store=store):
        store.search("before change")
        if change == "authorization":
            source.return_value = (replace(GRANT, authorization_fingerprint="b" * 64),)
        elif change == "agent_set":
            source.return_value = (replace(GRANT, agent_refs=frozenset({"7", "8"})),)
        elif change == "revoke":
            source.return_value = ()
        elif change == "source_error":
            source.side_effect = RuntimeError("synthetic verification unavailable")
        elif change == "bad_grant":
            source.return_value = (
                replace(GRANT, authorization_fingerprint="unverified"),
            )
        elif change == "lease_gap":
            publisher._deadline = time.monotonic() - 1
        else:
            reader._fresh_until = time.monotonic() - 1
        publisher.refresh()
        reader.refresh()
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("changed authority or broken continuity")
        assert transport.request.call_count == 1
        source.side_effect = None
        source.return_value = (GRANT,)
        assert publisher.refresh() and reader.refresh()
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("restored values cannot restore old scope")
        with delegated_agent_identity(7, store=store):
            store.search("freshly authorized scope")
        assert transport.request.call_count == 2


@pytest.mark.parametrize("change", ["revoke", "permissions"])
def test_missed_revoke_and_restore_still_fences_scope(live: Any, change: str) -> None:
    _path, _journal, source, publisher, reader, store, _transport = live
    with delegated_agent_identity(7, store=store):
        source.return_value = (
            ()
            if change == "revoke"
            else (replace(GRANT, authorization_fingerprint="b" * 64),)
        )
        assert publisher.refresh()
        source.return_value = (GRANT,)
        assert publisher.refresh() and reader.refresh()
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("reader missed the empty intermediate bundle")


def test_slow_verification_cannot_issue_a_fresh_full_lease(
    live: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, _journal, source, publisher, reader, store, _transport = live
    clock = [time.time(), time.monotonic()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "monotonic", lambda: clock[1])

    def slow() -> tuple[VerifiedSeatGrant, ...]:
        clock[0] += 61
        clock[1] += 61
        return (GRANT,)

    with delegated_agent_identity(7, store=store):
        source.side_effect = slow
        assert not publisher.refresh()
        # A concurrently scheduled refresh may also expire its verification;
        # admitting an empty publication or denying it must both deny the grant.
        reader.refresh()
        assert reader(REQUEST) is None
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("verification result is already too old")


def test_restart_cannot_accept_old_unexpired_bundle(live: Any) -> None:
    path, _journal, source, publisher, _reader, _store, _transport = live
    old = path.read_bytes()
    source.return_value = ()
    assert publisher.refresh()
    path.write_bytes(old)  # restore an actually valid, still-unexpired old bundle
    restarted = FileAgentSeatProvider(path)
    assert not restarted.refresh()  # nonce confirms a different current digest
    assert restarted(REQUEST) is None


def test_running_reader_cannot_miss_revoke_due_to_file_rollback(live: Any) -> None:
    path, _journal, source, publisher, reader, store, _transport = live
    old = path.read_bytes()
    with delegated_agent_identity(7, store=store):
        source.return_value = ()
        assert publisher.refresh()
        # Reader never observed the newer revision; its local floor alone
        # cannot detect this rollback. The fresh publisher challenge can.
        path.write_bytes(old)
        assert not reader.refresh()
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("old file hides the intermediate revocation")


def test_no_live_publisher_means_no_startup_from_disk(live: Any) -> None:
    path, _journal, _source, publisher, _reader, _store, _transport = live
    publisher.close()
    restarted = FileAgentSeatProvider(path)
    assert not restarted.refresh()
    assert restarted(REQUEST) is None


def test_captured_confirmation_cannot_answer_new_challenge(
    live: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _journal, _source, publisher, _reader, _store, _transport = live
    captured = publisher._acknowledge("b" * 64)
    monkeypatch.setattr(publisher, "_acknowledge", lambda _nonce: captured)
    assert not FileAgentSeatProvider(path).refresh()


def test_publisher_restart_reverifies_source_and_preserves_high_water(
    live: Any,
) -> None:
    path, journal, source, publisher, reader, store, _transport = live
    old_revision = json.loads(path.read_bytes())["revision"]
    with delegated_agent_identity(7, store=store):
        publisher.close()
        source.return_value = ()
        restarted = TrustedSeatPublisher(path, journal, source)
        restarted.start()
        try:
            assert json.loads(path.read_bytes())["revision"] > old_revision
            assert reader.refresh()
            assert reader(REQUEST) is None
            with pytest.raises(RemoteMemoryStoreIdentityError):
                store.search("publisher restart does not restore saved grants")
            source.return_value = (GRANT,)
            assert restarted.refresh() and reader.refresh()
            with pytest.raises(RemoteMemoryStoreIdentityError):
                store.search("old scope stays invalid")
        finally:
            restarted.close()


def test_missing_journal_does_not_reset_revision(live: Any) -> None:
    path, journal, source, publisher, _reader, _store, _transport = live
    publisher.close()
    journal.unlink()
    with pytest.raises(ValueError, match="journal missing"):
        TrustedSeatPublisher(path, journal, source).start()


def test_rolled_back_journal_cannot_overwrite_a_newer_projection(live: Any) -> None:
    path, journal, source, publisher, _reader, _store, _transport = live
    old = journal.read_bytes()
    assert publisher.refresh()
    publisher.close()
    journal.write_bytes(old)
    with pytest.raises(ValueError, match="behind current"):
        TrustedSeatPublisher(path, journal, source).start()


def test_only_one_publisher_owns_current(live: Any) -> None:
    path, journal, source, _publisher, _reader, _store, _transport = live
    with pytest.raises(BlockingIOError):
        TrustedSeatPublisher(path, journal, source).start()


def test_multiple_real_processes_renew_revoke_and_restart(live: Any) -> None:
    path, _journal, source, publisher, _reader, _store, _transport = live
    script = """
import os
import sys
from pathlib import Path
from xagent.core.memory.agent_seats import AgentSeatRequest, set_trusted_agent_seat_provider
from xagent.core.memory.remote import RemoteMemoryStore, RemoteMemoryStoreIdentityError, delegated_agent_identity
from xagent.core.memory.seat_file_provider import FileAgentSeatProvider
from xagent.core.user_context import current_user_id
reader = FileAgentSeatProvider(Path(sys.argv[1]))
request = AgentSeatRequest(sys.argv[2], "31", sys.argv[3])
os.environ.update(XAGENT_MEMORY_SERVICE_URL=sys.argv[2], XAGENT_MEMORY_SERVICE_KEY="synthetic-runtime-only", XAGENT_MEMORY_AGENT_SEATS="true")
current_user_id.set(31)
reader.refresh()
set_trusted_agent_seat_provider(reader)
store = RemoteMemoryStore(sys.argv[2], "synthetic-runtime-only")
scope = delegated_agent_identity(7, store=store)
try:
    scope.__enter__()
    entered = True
except RemoteMemoryStoreIdentityError:
    entered = False
for line in sys.stdin:
    reader.refresh()
    try:
        if not entered:
            raise RemoteMemoryStoreIdentityError("no initial grant")
        store.validate_identity_context()
        print(reader(request).generation, flush=True)
    except RemoteMemoryStoreIdentityError:
        print(0, flush=True)
if entered:
    scope.__exit__(None, None, None)
reader.close()
"""
    processes: list[subprocess.Popen[str]] = []

    def start() -> subprocess.Popen[str]:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                script,
                str(path),
                URL,
                REQUEST.runtime_profile,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        processes.append(process)
        return process

    def read(process: subprocess.Popen[str]) -> int:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write("refresh\n")
        process.stdin.flush()
        return int(process.stdout.readline())

    try:
        first, second = start(), start()
        generations = read(first), read(second)
        for _ in range(3):
            assert publisher.refresh()
            assert (read(first), read(second)) == generations
        source.return_value = ()
        assert publisher.refresh()
        assert (read(first), read(second), read(start())) == (0, 0, 0)
        source.return_value = (GRANT,)
        assert publisher.refresh()
        assert (read(first), read(second)) == (0, 0)
        assert read(start()) > 0
    finally:
        for process in processes:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
