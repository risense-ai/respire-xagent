"""Local publication and per-request validity for delegated identities."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import requests

from xagent.core.memory import agent_seats
from xagent.core.memory.agent_seats import (
    TrustedAgentSeatRegistry,
    set_trusted_agent_seat_provider,
)
from xagent.core.memory.remote import (
    RemoteMemoryStore,
    RemoteMemoryStoreIdentityError,
    delegated_agent_identity,
)
from xagent.core.user_context import current_user_id

URL = "http://memsrv:8080"
KEY = "runtime-memory-read-write-key"
PROFILE = hashlib.sha256(KEY.encode()).hexdigest()


class Transport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"results": []}'
        return response


@pytest.fixture
def publication(monkeypatch: pytest.MonkeyPatch):
    clock = [100.0]
    monkeypatch.setattr(agent_seats.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", URL)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", KEY)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "1")
    registry = TrustedAgentSeatRegistry()
    monkeypatch.setattr(agent_seats, "_provider", registry)
    monkeypatch.setattr(agent_seats, "_provider_generation", 1)
    snapshot = registry.publish(
        service_url=URL,
        user_id=31,
        runtime_profile=PROFILE,
        agent_ids=[7],
        ttl_seconds=30,
        status="active",
    )
    transport = Transport()
    store = RemoteMemoryStore(URL, KEY, session=transport)
    token = current_user_id.set(31)
    try:
        yield registry, snapshot, transport, store, clock
    finally:
        current_user_id.reset(token)


@pytest.mark.parametrize(
    "change",
    [
        "expiry",
        "user",
        "url",
        "key",
        "adapter_key",
        "mode",
        "revoke",
        "republish",
        "replace_provider",
        "clear_provider",
    ],
)
def test_active_scope_revalidates_before_each_request(
    publication, monkeypatch: pytest.MonkeyPatch, change: str
):
    registry, snapshot, transport, store, clock = publication
    with delegated_agent_identity(7, store=store):
        store.search("first valid request")
        if change == "expiry":
            clock[0] = snapshot.valid_until
        elif change == "user":
            current_user_id.set(32)
        elif change == "url":
            monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://other-service")
        elif change == "key":
            monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "other-runtime-key")
        elif change == "adapter_key":
            monkeypatch.setattr(store, "_service_key", "other-runtime-key")
        elif change == "mode":
            monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "0")
        elif change == "revoke":
            registry.revoke(service_url=URL, user_id=31, runtime_profile=PROFILE)
        elif change == "republish":
            registry.publish(
                service_url=URL,
                user_id=31,
                runtime_profile=PROFILE,
                agent_ids=[7],
                ttl_seconds=30,
                status="active",
            )
        elif change == "replace_provider":
            set_trusted_agent_seat_provider(registry)
        else:
            set_trusted_agent_seat_provider(None)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("must not dispatch")
    assert len(transport.calls) == 1
    assert transport.calls[0]["headers"]["X-User-External-Id"] == "agent:7"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: replace(s, status="disabled"),
        lambda s: replace(s, request=replace(s.request, user_external_id="32")),
        lambda s: replace(s, request=replace(s.request, runtime_profile="0" * 64)),
        lambda s: replace(s, agent_refs=frozenset({"8"})),
        lambda s: replace(s, generation=s.generation + 1),
        lambda s: replace(s, valid_until=s.valid_until + 30),
        lambda s: replace(s, valid_until=float("nan")),
    ],
)
def test_provider_result_changes_revoke_existing_binding(publication, mutation):
    _registry, snapshot, transport, store, _clock = publication
    with delegated_agent_identity(7, store=store):
        store.search("first")
        # Preserve installation epoch to exercise snapshot validation itself.
        agent_seats._provider = lambda _request: mutation(snapshot)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("must not dispatch")
    assert len(transport.calls) == 1


def test_republication_requires_new_scope_and_nested_scope_restores_parent(publication):
    registry, _snapshot, transport, store, _clock = publication
    with delegated_agent_identity(7, store=store):
        registry.publish(
            service_url=URL,
            user_id=31,
            runtime_profile=PROFILE,
            agent_ids=[7],
            ttl_seconds=30,
            status="active",
        )
        with delegated_agent_identity(7, store=store):
            store.search("fresh binding")
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("old binding restored and rejected")
    store.search("parent scope restored")
    assert [call["headers"]["X-User-External-Id"] for call in transport.calls] == [
        "agent:7",
        "31",
    ]


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_control_plane_publishes_for_distinct_restricted_runtime_credential(
    monkeypatch: pytest.MonkeyPatch,
):
    """Real loopback HTTP proves control credentials never enter the adapter.

    The fixture models the host's separate provisioning flow. This is not a
    production provider installer or an authorization substitute for memsrv.
    """
    calls = []
    control_key = "separate-control-plane-key"
    users = [
        {"kind": "person", "id": 61, "external_id": "31", "status": "active"},
        {"kind": "agent", "agent_ref": "7", "agent_owner_id": 61, "status": "active"},
        {"kind": "agent", "agent_ref": "8", "agent_owner_id": 61, "status": "disabled"},
        {"kind": "agent", "agent_ref": "9", "agent_owner_id": 99, "status": "active"},
    ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, code, payload):
            content = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            auth = self.headers.get("Authorization")
            calls.append(("control", auth, self.headers.get("X-User-External-Id")))
            self.reply(
                200 if auth == f"Bearer {control_key}" else 403, {"users": users}
            )

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            auth, identity = (
                self.headers.get("Authorization"),
                self.headers.get("X-User-External-Id"),
            )
            calls.append(("memory", auth, identity))
            self.reply(
                200 if auth == f"Bearer {KEY}" and identity == "agent:7" else 403,
                {"results": []},
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    registry = TrustedAgentSeatRegistry()
    monkeypatch.setattr(agent_seats, "_provider", registry)
    monkeypatch.setattr(agent_seats, "_provider_generation", 1)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", url)
    monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", KEY)
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "1")
    control = requests.Session()
    runtime = requests.Session()
    control.trust_env = runtime.trust_env = False
    token = current_user_id.set(31)
    try:
        rows = control.get(
            url + "/v1/org/users",
            headers={"Authorization": f"Bearer {control_key}"},
            timeout=2,
        ).json()["users"]
        # The trusted publisher verifies status and ownership; org membership
        # by itself is deliberately insufficient to publish a user grant.
        person = next(
            row
            for row in rows
            if row.get("kind") == "person"
            and row.get("external_id") == "31"
            and row.get("status") == "active"
        )
        refs = [
            int(row["agent_ref"])
            for row in rows
            if row.get("kind") == "agent"
            and row.get("status") == "active"
            and row.get("agent_owner_id") == person["id"]
        ]
        # Publishing under the admin credential's profile cannot authorize
        # this runtime. The host must name the runtime profile explicitly.
        registry.publish(
            service_url=url,
            user_id=31,
            runtime_profile=hashlib.sha256(control_key.encode()).hexdigest(),
            agent_ids=refs,
            ttl_seconds=30,
            status="active",
        )
        store = RemoteMemoryStore(url, KEY, session=runtime)
        with (
            pytest.raises(RemoteMemoryStoreIdentityError),
            delegated_agent_identity(7, store=store),
        ):
            pytest.fail("control credential profile leaked into runtime")
        registry.publish(
            service_url=url,
            user_id=31,
            runtime_profile=PROFILE,
            agent_ids=refs,
            ttl_seconds=30,
            status="active",
        )
        store = RemoteMemoryStore(url, KEY, session=runtime)
        with delegated_agent_identity(7, store=store):
            assert store.search("published seat") == []
        for denied_agent in [8, 9]:
            with (
                pytest.raises(RemoteMemoryStoreIdentityError),
                delegated_agent_identity(denied_agent, store=store),
            ):
                pytest.fail("disabled or foreign seat entered")
        assert calls == [
            ("control", f"Bearer {control_key}", None),
            ("memory", f"Bearer {KEY}", "agent:7"),
        ]
    finally:
        current_user_id.reset(token)
        control.close()
        runtime.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize("status", ["disabled", "pending", "", None])
def test_publisher_requires_verified_active_status(status):
    registry = TrustedAgentSeatRegistry()
    with pytest.raises(ValueError):
        registry.publish(
            service_url=URL,
            user_id=31,
            runtime_profile=PROFILE,
            agent_ids=[7],
            ttl_seconds=30,
            status=status,
        )


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("change", ["mode", "user", "url", "key", "adapter_key"])
def test_both_identity_modes_bind_entry_context_until_scope_exit(
    publication, monkeypatch, independent, change
):
    _registry, _snapshot, transport, store, _clock = publication
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "1" if independent else "0")
    with delegated_agent_identity(7, store=store):
        store.search("valid entered scope")
        if change == "mode":
            monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "0" if independent else "1")
        elif change == "user":
            current_user_id.set(32)
        elif change == "url":
            monkeypatch.setenv("XAGENT_MEMORY_SERVICE_URL", "http://other-service")
        elif change == "key":
            monkeypatch.setenv("XAGENT_MEMORY_SERVICE_KEY", "other-runtime-key")
        else:
            monkeypatch.setattr(store, "_service_key", "other-runtime-key")
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("changed context cannot reuse old scope")
    assert len(transport.calls) == 1
    assert transport.calls[0]["headers"]["X-User-External-Id"] == (
        "agent:7" if independent else "31"
    )


def test_explicit_parent_mode_needs_no_provider_and_resolves_a_fresh_scope_after_change(
    publication, monkeypatch
):
    _registry, _snapshot, transport, store, _clock = publication
    monkeypatch.setenv("XAGENT_MEMORY_AGENT_SEATS", "0")
    set_trusted_agent_seat_provider(None)
    with delegated_agent_identity(7, store=store):
        store.search("parent request")
        current_user_id.set(32)
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("old parent user")
        with delegated_agent_identity(7, store=store):
            store.search("new parent user explicitly resolved")
        with pytest.raises(RemoteMemoryStoreIdentityError):
            store.search("restored old parent binding remains invalid")
    assert [call["headers"]["X-User-External-Id"] for call in transport.calls] == [
        "31",
        "32",
    ]
