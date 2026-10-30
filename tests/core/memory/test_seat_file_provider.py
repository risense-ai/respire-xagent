"""Synthetic host publications: no governance credential or memory service I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import requests

from xagent import config
from xagent.core.memory import agent_seats
from xagent.core.memory import seat_file_provider as module
from xagent.core.memory.agent_seats import AgentSeatRequest
from xagent.core.memory.remote import (
    RemoteMemoryStore,
    RemoteMemoryStoreIdentityError,
    delegated_agent_identity,
)
from xagent.core.user_context import current_user_id

URL = "http://memsrv:8080"
KEY = "synthetic-runtime-key"
PROFILE = hashlib.sha256(KEY.encode()).hexdigest()
REQUEST = AgentSeatRequest(URL, "31", PROFILE)


def bundle(revision: int = 1) -> dict[str, Any]:
    return {
        "version": 1,
        "revision": revision,
        "issued_at": 1_000,
        "expires_at": 1_100,
        "grants": [
            {
                "service_url": URL,
                "user_id": 31,
                "runtime_profile": PROFILE,
                "agent_ids": [7],
                "status": "active",
            }
        ],
    }


def publish(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(".next")
    temporary.write_text(json.dumps(document))
    temporary.chmod(0o600)
    temporary.replace(path)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    value = [1_000.0, 50.0]
    monkeypatch.setattr(module.time, "time", lambda: value[0])
    monkeypatch.setattr(module.time, "monotonic", lambda: value[1])
    return value


def test_unchanged_refresh_preserves_generation_and_does_not_extend_expiry(
    tmp_path: Path, clock: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "grants.json"
    publish(path, bundle())
    provider = module.FileAgentSeatProvider(path)
    assert provider.refresh()
    initial = provider(REQUEST)
    assert initial is not None and initial.valid_until == 150
    clock[0] += 1
    clock[1] += 1
    assert provider.refresh()
    assert provider(REQUEST) is initial
    monkeypatch.setattr(
        provider, "_read", lambda: pytest.fail("runtime performed file I/O")
    )
    assert provider(AgentSeatRequest(URL, "32", PROFILE)) is None
    assert provider(AgentSeatRequest(URL, "31", "0" * 64)) is None
    assert provider(AgentSeatRequest("http://other", "31", PROFILE)) is None
    assert provider(REQUEST) is initial


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "malformed",
        "symlink",
        "writable",
        "rollback",
        "same_revision_changed",
        "expired",
        "future",
        "lifetime",
        "duplicate_key",
        "oversized",
        "fifo",
    ],
)
def test_bad_source_revokes_and_requires_new_revision(
    tmp_path: Path, clock: list[float], change: str
) -> None:
    path = tmp_path / "grants.json"
    publish(path, bundle(2))
    provider = module.FileAgentSeatProvider(path)
    assert provider.refresh()
    document = bundle(2)
    if change == "missing":
        path.unlink()
    elif change == "malformed":
        path.write_text("{")
    elif change == "symlink":
        other = tmp_path / "other.json"
        publish(other, document)
        path.unlink()
        path.symlink_to(other)
    elif change == "writable":
        path.chmod(0o666)
    elif change == "duplicate_key":
        path.write_text('{"version": 1, "version": 1}')
    elif change == "oversized":
        path.write_text(" " * (module._MAX_BYTES + 1))
    elif change == "fifo":
        import os

        path.unlink()
        os.mkfifo(path)
    else:
        if change == "rollback":
            document["revision"] = 1
        elif change == "same_revision_changed":
            document["grants"][0]["agent_ids"] = [8]
        elif change == "expired":
            document["expires_at"] = 999
        elif change == "future":
            document["issued_at"] = 1001
        else:
            document["expires_at"] = 1_301
        publish(path, document)
    assert not provider.refresh()
    assert provider(REQUEST) is None
    if path.is_symlink() or change == "fifo":
        path.unlink()
    publish(path, bundle(2))
    assert not provider.refresh()
    publish(path, bundle(3))
    assert provider.refresh()
    assert provider(REQUEST) is not None


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "disabled",
        "bad_user",
        "bool_user",
        "bool_agent",
        "duplicate_agent",
        "duplicate_target",
        "bad_profile",
        "url_credential",
        "url_query",
        "nan",
        "bool_version",
    ],
)
def test_strict_grant_schema_rejects_whole_bundle(
    tmp_path: Path, clock: list[float], mutation: str
) -> None:
    document = bundle()
    grant = document["grants"][0]
    if mutation == "unknown":
        grant["control_key"] = "never-accepted"
    elif mutation == "disabled":
        grant["status"] = "disabled"
    elif mutation == "bad_user":
        grant["user_id"] = "31"
    elif mutation == "bool_user":
        grant["user_id"] = True
    elif mutation == "bool_agent":
        grant["agent_ids"] = [True]
    elif mutation == "duplicate_agent":
        grant["agent_ids"] = [7, 7]
    elif mutation == "duplicate_target":
        document["grants"].append(dict(grant))
    elif mutation == "bad_profile":
        grant["runtime_profile"] = "control-credential"
    elif mutation == "url_credential":
        grant["service_url"] = "http://user:secret@memsrv"
    elif mutation == "url_query":
        grant["service_url"] = URL + "?key=secret"
    elif mutation == "nan":
        document["expires_at"] = float("nan")
    else:
        document["version"] = True
    path = tmp_path / "grants.json"
    publish(path, document)
    provider = module.FileAgentSeatProvider(path)
    assert not provider.refresh()
    assert provider(REQUEST) is None


def test_refresh_stall_and_restart_do_not_renew_stale_authority(
    tmp_path: Path, clock: list[float]
) -> None:
    path = tmp_path / "grants.json"
    publish(path, bundle())
    provider = module.FileAgentSeatProvider(path)
    assert provider.refresh()
    clock[1] += 3
    assert provider(REQUEST) is None
    assert not provider.refresh()  # same revision cannot recover a stalled source
    publish(path, bundle(2))
    assert provider.refresh()
    provider.close()
    assert provider(REQUEST) is None
    assert not provider.refresh()
    clock[0] = 1_100
    restarted = module.FileAgentSeatProvider(path)
    assert not restarted.refresh()
    assert restarted(REQUEST) is None


def test_two_readers_observe_revoke_and_old_scope_never_revives(
    tmp_path: Path, clock: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "grants.json"
    publish(path, bundle())
    first, second = (
        module.FileAgentSeatProvider(path),
        module.FileAgentSeatProvider(path),
    )
    assert first.refresh() and second.refresh()
    monkeypatch.setattr(agent_seats, "_provider", first)
    monkeypatch.setattr(agent_seats, "_provider_generation", 1)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", URL)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", KEY)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "true")
    calls: list[dict[str, Any]] = []

    class Session(requests.Session):
        def request(self, *args: Any, **kwargs: Any) -> requests.Response:
            calls.append(kwargs)
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"results": []}'
            return response

    store = RemoteMemoryStore(URL, KEY, session=Session())
    token = current_user_id.set(31)
    try:
        with delegated_agent_identity(7, store=store):
            store.search("synthetic")
            revoke = bundle(2)
            revoke["grants"] = []
            publish(path, revoke)
            assert first.refresh() and second.refresh()
            assert first(REQUEST) is None and second(REQUEST) is None
            with pytest.raises(RemoteMemoryStoreIdentityError):
                store.search("revoked")
            publish(path, bundle(3))
            assert first.refresh() and second.refresh()
            with pytest.raises(RemoteMemoryStoreIdentityError):
                store.search("old scope")
            with delegated_agent_identity(7, store=store):
                store.search("fresh scope")
        assert len(calls) == 2
        assert all(c["headers"]["X-User-External-Id"] == "agent:7" for c in calls)
        assert all(c["headers"]["Authorization"] == f"Bearer {KEY}" for c in calls)
    finally:
        current_user_id.reset(token)


def test_runtime_installs_without_changing_mode_or_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_MEMORY_BACKEND", "remote")
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "true")
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setattr(agent_seats, "_provider", None)
    monkeypatch.setattr(agent_seats, "_provider_generation", 0)
    module.start_memory_seat_runtime()
    try:
        with pytest.raises(ValueError, match="invalid or expired"):
            agent_seats.read_agent_seat_snapshot(REQUEST)
        assert config.get_memory_agent_seats_enabled()
    finally:
        module.stop_memory_seat_runtime()
    with pytest.raises(ValueError):
        agent_seats.read_agent_seat_snapshot(REQUEST)


def test_real_processes_refresh_independently_and_restart_reads_latest_bundle(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys
    import time

    path = tmp_path / "grants.json"
    document = bundle()
    document.update(issued_at=time.time(), expires_at=time.time() + 120)
    publish(path, document)
    script = """
import sys
from pathlib import Path
from xagent.core.memory.agent_seats import AgentSeatRequest
from xagent.core.memory.seat_file_provider import FileAgentSeatProvider
provider = FileAgentSeatProvider(Path(sys.argv[1]))
request = AgentSeatRequest(sys.argv[2], "31", sys.argv[3])
provider.start()
for line in sys.stdin:
    if line.strip() == "stop":
        break
    print("yes" if provider(request) else "no", flush=True)
provider.close()
"""
    readers: list[subprocess.Popen[str]] = []

    def start() -> subprocess.Popen[str]:
        reader = subprocess.Popen(
            [sys.executable, "-u", "-c", script, str(path), URL, PROFILE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        readers.append(reader)
        return reader

    def read(reader: subprocess.Popen[str]) -> str:
        assert reader.stdin is not None and reader.stdout is not None
        reader.stdin.write("read\n")
        reader.stdin.flush()
        return reader.stdout.readline().strip()

    try:
        first, second = start(), start()
        assert read(first) == read(second) == "yes"
        document.update(revision=2, grants=[])
        publish(path, document)
        deadline = time.monotonic() + 4
        while True:
            answers = [read(first), read(second)]
            if answers == ["no", "no"]:
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        restarted = start()
        assert read(restarted) == "no"
    finally:
        for reader in readers:
            if reader.stdin is not None:
                reader.stdin.close()
            try:
                reader.wait(timeout=5)
            except subprocess.TimeoutExpired:
                reader.kill()
                reader.wait(timeout=2)
