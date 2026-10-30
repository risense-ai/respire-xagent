"""Slow real local HTTP must not block leased-state Unix challenges."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.core.memory import test_seat_publisher as publisher_fixtures
from tests.core.memory import test_seat_verifier as verifier_fixtures
from xagent.core.memory import agent_seats
from xagent.core.memory.agent_seats import AgentSeatRequest
from xagent.core.memory.remote import (
    RemoteMemoryStore,
    RemoteMemoryStoreIdentityError,
    delegated_agent_identity,
)
from xagent.core.memory.seat_file_provider import FileAgentSeatProvider
from xagent.core.memory.seat_publisher import TrustedSeatPublisher
from xagent.core.memory.seat_verifier import MemsrvSeatVerifier, UserBinding
from xagent.core.user_context import current_user_id

verifier = verifier_fixtures.verifier
live = publisher_fixtures.live


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1", "localhost", "::1"])
def test_slow_multiuser_http_keeps_parallel_challenges_and_scope_alive(
    verifier: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture_source, state, _calls, engine, control, runtime = verifier
    with engine.begin() as connection:
        connection.exec_driver_sql("INSERT INTO users VALUES (32,0)")
        connection.exec_driver_sql(
            "INSERT INTO agents VALUES (8,31,'published',NULL,'team'),(9,32,'published',NULL,'team'),(10,32,'published',NULL,'team')"
        )
    state["rows"].extend(
        [
            {
                "id": 202,
                "external_id": "32",
                "kind": "person",
                "status": "active",
                "role": "member",
            },
            *[
                {
                    "id": id_,
                    "agent_ref": str(ref),
                    "agent_owner_id": owner,
                    "kind": "agent",
                    "status": "active",
                    "role": "member",
                }
                for id_, ref, owner in ((203, 8, 200), (204, 9, 202), (205, 10, 202))
            ],
        ]
    )
    slow_started, verification_finished = threading.Event(), threading.Event()
    state["delay"] = 0.0
    durations: list[float] = []
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            delay = state["delay"]
            if delay:
                slow_started.set()
                time.sleep(delay)
            if self.path == "/v1/org/users":
                assert self.headers["Authorization"] == "Bearer synthetic-control"
                payload = {"users": state["rows"]}
            else:
                assert self.path == "/v1/whoami"
                assert self.headers["Authorization"] == "Bearer synthetic-runtime"
                external = self.headers["X-User-External-Id"]
                selected = next(
                    row
                    for row in state["rows"]
                    if row.get("external_id") == external
                    or external == "agent:" + str(row.get("agent_ref"))
                )
                payload = {
                    "identity": {"id": selected["id"], "role": "member"},
                    "kind": "service",
                    "is_admin": False,
                    "scopes": ["memory:read", "memory:write"],
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            return

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http_thread = threading.Thread(target=http.serve_forever, daemon=True)
    http_thread.start()
    url = f"http://127.0.0.1:{http.server_port}"
    source = MemsrvSeatVerifier(
        engine=engine,
        service_url=url,
        control_key_file=control,
        runtime_key_file=runtime,
        bindings=(UserBinding(31, 200), UserBinding(32, 202)),
    )

    def verify() -> Any:
        delayed = bool(state["delay"])
        start = time.monotonic()
        result = source()
        if delayed:
            durations.append(time.monotonic() - start)
            verification_finished.set()
        return result

    path = tmp_path / "grant.json"
    publisher = TrustedSeatPublisher(
        path, tmp_path / "state.json", verify, ttl_seconds=10, poll_seconds=0.1
    )
    publisher.start()
    main_reader = FileAgentSeatProvider(path)
    assert main_reader.refresh()
    request = AgentSeatRequest(
        url, "31", hashlib.sha256(b"synthetic-runtime").hexdigest()
    )
    initial = main_reader(request)
    assert initial is not None
    initial_revision = json.loads(path.read_bytes())["revision"]
    monkeypatch.setattr(agent_seats, "_provider", main_reader)
    monkeypatch.setattr(agent_seats, "_provider_generation", 1)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", url)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "synthetic-runtime")
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "true")
    store = RemoteMemoryStore(url, "synthetic-runtime")
    token = current_user_id.set(31)
    try:
        with delegated_agent_identity(7, store=store):
            state["delay"] = 0.2  # eight actual HTTP requests: >=1.6s verification
            assert slow_started.wait(2)
            assert main_reader.refresh()
            assert store.validate_identity_context() == "agent:7"

            def challenge() -> float:
                reader = FileAgentSeatProvider(path)
                start = time.monotonic()
                assert reader.refresh()
                assert reader(request) is not None
                reader.close()
                return time.monotonic() - start

            with ThreadPoolExecutor(max_workers=4) as pool:
                observed = list(pool.map(lambda _index: challenge(), range(12)))
            assert max(observed) < 0.8  # existing reader socket timeout stays 1s
            deadline = time.monotonic() + 5
            while not verification_finished.is_set():
                assert time.monotonic() < deadline
                assert main_reader.refresh()
                assert store.validate_identity_context() == "agent:7"
                time.sleep(0.04)
            state["delay"] = 0.0
            while json.loads(path.read_bytes())["revision"] <= initial_revision:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert main_reader.refresh()
            assert store.validate_identity_context() == "agent:7"
            renewed = main_reader(request)
            assert renewed is not None and renewed.generation == initial.generation
            assert min(durations) > 1.2
            assert len(paths) >= 16  # bootstrap plus actual slow multi-user recheck
    finally:
        state["delay"] = 0.0
        current_user_id.reset(token)
        main_reader.close()
        publisher.close()
        source.session.close()
        http.shutdown()
        http.server_close()
        http_thread.join(timeout=2)


@pytest.mark.parametrize("phase", ["journal", "manifest_prepare"])
def test_durable_preparation_does_not_hold_challenge_lock(
    live: Any, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    path, journal, _source, publisher, reader, store, _transport = live
    entered, release = threading.Event(), threading.Event()
    method = "_atomic_write" if phase == "journal" else "_prepare_write"
    original = getattr(publisher, method)

    def delayed(target: Path, raw: bytes) -> Any:
        if target == (journal if phase == "journal" else path):
            entered.set()
            assert release.wait(3)
        return original(target, raw)

    monkeypatch.setattr(publisher, method, delayed)
    with delegated_agent_identity(7, store=store):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(publisher.refresh)
            try:
                assert entered.wait(1)
                start = time.monotonic()
                assert reader.refresh()
                assert time.monotonic() - start < 0.8
                assert store.validate_identity_context() == "agent:7"
            finally:
                release.set()
            assert pending.result(timeout=2)
        assert reader.refresh()
        assert store.validate_identity_context() == "agent:7"


def test_expiry_is_not_extended_while_verification_is_blocked(
    live: Any,
) -> None:
    _path, _journal, source, publisher, reader, store, _transport = live
    entered, release = threading.Event(), threading.Event()

    def blocked() -> Any:
        entered.set()
        assert release.wait(3)
        return (publisher_fixtures.GRANT,)

    with delegated_agent_identity(7, store=store):
        source.side_effect = blocked
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(publisher.refresh)
            try:
                assert entered.wait(1)
                publisher._deadline = time.monotonic() - 1
                start = time.monotonic()
                assert not reader.refresh()
                assert time.monotonic() - start < 0.8
                with pytest.raises(RemoteMemoryStoreIdentityError):
                    store.validate_identity_context()
            finally:
                release.set()
            assert pending.result(timeout=2)
        assert reader.refresh()
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.validate_identity_context()
        with delegated_agent_identity(7, store=store):
            assert store.validate_identity_context() == "agent:7"
