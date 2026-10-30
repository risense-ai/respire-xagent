"""Consume host-verified grants without giving runtime a control credential.

The configured directory must be a read-only bind mount owned by the trusted
publisher, outside uploads/workspaces and agent-writable storage. This is a
trusted local input, not a signed document or a provisioning service. The host
must verify user/seat ownership and memsrv authorization before publishing.

Publish via atomic rename inside the mounted directory (not a single-file bind
mount, which can pin an old inode). Each replacement, including recovery after
an observed error, needs a larger revision. Readers poll once per second; local
calls do no file/network I/O and fail closed if refresh stalls for three seconds.
Absolute expiries cannot be extended by rereading or restarting a worker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ...config import get_memory_agent_seats_file, get_memory_backend
from .agent_seats import (
    AgentSeatRequest,
    AgentSeatSnapshot,
    set_trusted_agent_seat_provider,
)

logger = logging.getLogger(__name__)
_MAX_BYTES = 1_048_576
_MAX_TTL = 300
_POLL_SECONDS = 1.0
_FRESH_SECONDS = 3.0


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate grant field")
        result[key] = value
    return result


class FileAgentSeatProvider:
    """Per-process immutable publications, refreshed by a separate host thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._revision = 0
        self._digest: bytes | None = None
        self._snapshots: dict[AgentSeatRequest, AgentSeatSnapshot] = {}
        self._fresh_until = 0.0
        self._healthy = False
        self._closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __call__(self, request: AgentSeatRequest) -> AgentSeatSnapshot | None:
        with self._lock:
            if self._closed or time.monotonic() >= self._fresh_until:
                self._invalidate()
                return None
            snapshot = self._snapshots.get(request)
            if snapshot is None or time.monotonic() >= snapshot.valid_until:
                return None
            return snapshot

    def _invalidate(self) -> None:
        # Keep revision/digest as a recovery fence. Replaying the same bundle
        # after a read failure cannot revive scopes that missed the failure.
        self._snapshots = {}
        self._healthy = False
        self._fresh_until = 0.0

    def _read(self) -> bytes:
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
        fd = os.open(self.path, flags)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o022:
                raise ValueError(
                    "grant file must be regular and not group/world writable"
                )
            raw = source.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            raise ValueError("grant file too large")
        return raw

    def refresh(self) -> bool:
        """Host-side disk refresh. Errors revoke; no payload/path is logged."""
        try:
            raw = self._read()
            document = json.loads(raw, object_pairs_hook=_unique_object)
            now, mono = time.time(), time.monotonic()
            revision, snapshots = self._parse(document, now, mono)
            digest = hashlib.sha256(raw).digest()
            with self._lock:
                if self._closed:
                    return False
                if revision < self._revision or (
                    revision == self._revision
                    and (not self._healthy or digest != self._digest)
                ):
                    raise ValueError("grant revision rollback or replay")
                if revision > self._revision:
                    self._snapshots = snapshots
                    self._revision, self._digest = revision, digest
                # Identical rereads preserve monotonic expiry and generation.
                self._healthy = True
                self._fresh_until = mono + _FRESH_SECONDS
            return True
        except (OSError, ValueError, TypeError, OverflowError, RecursionError):
            with self._lock:
                was_healthy = self._healthy
                self._invalidate()
            if was_healthy:
                logger.warning(
                    "Memory seat publication unavailable; independent seats denied"
                )
            return False

    @staticmethod
    def _parse(
        document: Any, now: float, mono: float
    ) -> tuple[int, dict[AgentSeatRequest, AgentSeatSnapshot]]:
        if (
            type(document) is not dict
            or set(document)
            != {"version", "revision", "issued_at", "expires_at", "grants"}
            or type(document["version"]) is not int
            or document["version"] != 1
            or not _positive_int(document["revision"])
            or not _finite_number(document["issued_at"])
            or not _finite_number(document["expires_at"])
            or not document["issued_at"] <= now < document["expires_at"]
            or not 0 < document["expires_at"] - document["issued_at"] <= _MAX_TTL
            or type(document["grants"]) is not list
            or len(document["grants"]) > 10_000
        ):
            raise ValueError("invalid grant bundle")
        snapshots: dict[AgentSeatRequest, AgentSeatSnapshot] = {}
        for grant in document["grants"]:
            if type(grant) is not dict or set(grant) != {
                "service_url",
                "user_id",
                "runtime_profile",
                "agent_ids",
                "status",
            }:
                raise ValueError("invalid grant")
            url, profile, agents = (
                grant["service_url"],
                grant["runtime_profile"],
                grant["agent_ids"],
            )
            if (
                type(url) is not str
                or url != url.strip().rstrip("/")
                or urlsplit(url).scheme not in {"http", "https"}
                or not urlsplit(url).netloc
                or urlsplit(url).username is not None
                or urlsplit(url).query
                or urlsplit(url).fragment
                or not _positive_int(grant["user_id"])
                or type(profile) is not str
                or len(profile) != 64
                or any(c not in "0123456789abcdef" for c in profile)
                or type(agents) is not list
                or not all(_positive_int(agent) for agent in agents)
                or len(set(agents)) != len(agents)
                or grant["status"] != "active"
            ):
                raise ValueError("invalid grant target or attestation")
            request = AgentSeatRequest(url, str(grant["user_id"]), profile)
            if request in snapshots:
                raise ValueError("duplicate grant target")
            snapshots[request] = AgentSeatSnapshot(
                request,
                frozenset(str(agent) for agent in agents),
                document["revision"],
                mono + document["expires_at"] - now,
                "active",
            )
        return document["revision"], snapshots

    def start(self) -> None:
        if not self.refresh():
            logger.warning("Memory seat file not admitted; independent seats denied")
        self._thread = threading.Thread(
            target=self._run, name="memory-seat-publications", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(_POLL_SECONDS):
            self.refresh()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._invalidate()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


_runtime: FileAgentSeatProvider | None = None


def start_memory_seat_runtime() -> None:
    """Call in each web/shared worker or Celery execution process, after fork."""
    global _runtime
    stop_memory_seat_runtime()
    if get_memory_backend() != "remote":
        return
    path = get_memory_agent_seats_file()
    if path is None:
        # Custom host-installed providers remain supported. No file means no
        # built-in supply, not permission to use another acting identity.
        return
    provider = FileAgentSeatProvider(path)
    _runtime = provider
    set_trusted_agent_seat_provider(provider)
    provider.start()


def stop_memory_seat_runtime() -> None:
    global _runtime
    runtime, _runtime = _runtime, None
    if runtime is not None:
        runtime.close()
