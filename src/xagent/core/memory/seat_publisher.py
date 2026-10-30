"""Host-side publisher mechanism; no built-in governance connector/credential.

The host supplies a verifier returning only explicitly authorized test grants.
It must attest ownership, status, identity selection and all relevant policy in
an authorization fingerprint. Runtime processes never invoke that verifier.
The socket/current directory is host-controlled and read-only in containers;
the journal belongs to the publisher and must survive process/container restarts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import socketserver
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_seats import AgentSeatRequest
from .seat_file_provider import _MAX_TTL, FileAgentSeatProvider, _unique_object

logger = logging.getLogger(__name__)


class GrantVerificationError(ValueError):
    """Controlled operator reason without response bodies, URLs or credentials."""


@dataclass(frozen=True)
class VerifiedSeatGrant:
    request: AgentSeatRequest
    agent_refs: frozenset[str]
    authorization_fingerprint: str


class TrustedSeatPublisher:
    """Single host writer with fresh startup, persistent revision and local IPC.

    A restart never serves the saved bundle: it verifies current authorization
    before listening. New publisher IDs fence existing scopes on restart even
    when grants are unchanged. The journal is a crash-recovery high-water mark,
    not a defense against a malicious rollback of the whole trusted host. Fresh
    verification plus the reader's nonce confirmation protects a restarted
    reader from stale files even if its process-local revision starts at zero.
    """

    def __init__(
        self,
        path: Path,
        journal: Path,
        verifier: Callable[[], Iterable[VerifiedSeatGrant]],
        *,
        ttl_seconds: float = 60,
        poll_seconds: float = 10,
    ) -> None:
        if not path.is_absolute() or not journal.is_absolute() or path == journal:
            raise ValueError(
                "publisher needs separate absolute manifest and journal paths"
            )
        if not 0 < poll_seconds < ttl_seconds <= _MAX_TTL:
            raise ValueError("invalid publisher lifetime or refresh interval")
        self.path, self.journal, self.verifier = path, journal, verifier
        self.ttl_seconds, self.poll_seconds = ttl_seconds, poll_seconds
        self.publisher_id = uuid.uuid4().hex
        self._revision = 0
        self._current: dict[str, Any] | None = None
        self._digest = ""
        self._deadline = 0.0
        self._authorities: dict[AgentSeatRequest, tuple[VerifiedSeatGrant, int]] = {}
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._server: socketserver.UnixStreamServer | None = None
        self._thread: threading.Thread | None = None
        self._verification_thread: threading.Thread | None = None
        self._writer_fd: int | None = None

    @staticmethod
    def _bytes(document: Any) -> bytes:
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    @staticmethod
    def _prepare_write(path: Path, raw: bytes) -> Path:
        temporary = path.with_name(path.name + ".next")
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        return temporary

    @staticmethod
    def _sync_directory(path: Path) -> None:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @classmethod
    def _atomic_write(cls, path: Path, raw: bytes) -> None:
        temporary = cls._prepare_write(path, raw)
        temporary.replace(path)
        cls._sync_directory(path)

    def _acquire_writer(self) -> None:
        lock_path = self.path.with_name(self.path.name + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.journal.exists():
                saved = json.loads(
                    self.journal.read_bytes(), object_pairs_hook=_unique_object
                )
                document = saved["current"]
                if (
                    type(document["revision"]) is not int
                    or document["revision"] <= 0
                    or hashlib.sha256(self._bytes(document)).hexdigest()
                    != saved["sha256"]
                ):
                    raise ValueError("invalid publication journal")
                self._revision = document["revision"]
                if self.path.exists():
                    projected = json.loads(self.path.read_bytes())
                    if projected["revision"] > self._revision:
                        raise ValueError("publication journal is behind current file")
            elif self.path.exists():
                raise ValueError(
                    "publication journal missing; do not replay an existing file"
                )
            self._writer_fd = fd
        except BaseException:
            os.close(fd)
            raise

    def refresh(self) -> bool:
        """Reverify complete authorization. Any verifier failure revokes all."""
        # Only verification/publication writers serialize here. Never hold the
        # challenge-state lock over SQL, HTTP, serialization or durable writes.
        with self._refresh_lock:
            if self._writer_fd is None:
                raise RuntimeError("publisher not started")
            if self._stop.is_set():
                return False
            verified_at = time.time(), time.monotonic()
            try:
                grants = tuple(self.verifier())
            except Exception as error:  # noqa: BLE001 - host verifier failures must revoke
                grants = ()
                self._invalidate_publication()
                reason = (
                    str(error)
                    if isinstance(error, GrantVerificationError)
                    else type(error).__name__
                )
                logger.warning(
                    "Memory grant verification failed (%s); publishing no grants",
                    reason,
                )
            try:
                self._publish(grants, verified_at=verified_at)
            except (ValueError, TypeError, OverflowError, AttributeError):
                self._invalidate_publication()
                if self._stop.is_set():
                    return False
                try:
                    self._publish(())
                except (OSError, ValueError):
                    return False
                return False
            except OSError:
                self._invalidate_publication()
                logger.warning("Memory grant publication unavailable")
                return False
            return True

    def _invalidate_publication(self) -> None:
        with self._lock:
            self._current = None
            self._authorities = {}

    def _publish(
        self,
        grants: tuple[VerifiedSeatGrant, ...],
        *,
        verified_at: tuple[float, float] | None = None,
    ) -> None:
        if self._stop.is_set():
            raise ValueError("publisher stopped")
        now, mono = time.time(), time.monotonic()
        issued_at, verified_mono = verified_at or (now, mono)
        if mono - verified_mono >= self.ttl_seconds:
            raise ValueError("verification exceeded its lease lifetime")
        revision = self._revision + 1
        continuous = self._current is not None and mono < self._deadline
        authorities: dict[AgentSeatRequest, tuple[VerifiedSeatGrant, int]] = {}
        rows = []
        for grant in grants:
            if (
                not isinstance(grant, VerifiedSeatGrant)
                or type(grant.authorization_fingerprint) is not str
                or len(grant.authorization_fingerprint) != 64
                or any(
                    c not in "0123456789abcdef" for c in grant.authorization_fingerprint
                )
                or type(grant.agent_refs) is not frozenset
                or not grant.request.user_external_id.isascii()
                or not grant.request.user_external_id.isdecimal()
                or str(int(grant.request.user_external_id))
                != grant.request.user_external_id
                or int(grant.request.user_external_id) <= 0
                or grant.request in authorities
                or any(
                    not ref.isascii()
                    or not ref.isdecimal()
                    or str(int(ref)) != ref
                    or int(ref) <= 0
                    for ref in grant.agent_refs
                )
            ):
                raise ValueError("invalid verified authorization")
            previous = self._authorities.get(grant.request)
            epoch = (
                previous[1]
                if continuous and previous is not None and previous[0] == grant
                else revision
            )
            authorities[grant.request] = grant, epoch
            rows.append(
                {
                    "service_url": grant.request.service_url,
                    "user_id": int(grant.request.user_external_id),
                    "runtime_profile": grant.request.runtime_profile,
                    "agent_ids": sorted(int(ref) for ref in grant.agent_refs),
                    "status": "active",
                    "authorization_epoch": epoch,
                }
            )
        document = {
            "version": 2,
            "revision": revision,
            "publisher_id": self.publisher_id,
            "issued_at": issued_at,
            "expires_at": issued_at + self.ttl_seconds,
            "grants": rows,
        }
        FileAgentSeatProvider._parse(document, now, mono)
        raw = self._bytes(document)
        saved = self._bytes(
            {"current": document, "sha256": hashlib.sha256(raw).hexdigest()}
        )
        # Reserve the new high-water mark before replacing the projection.
        self._atomic_write(self.journal, saved)
        self._revision = revision
        temporary = self._prepare_write(self.path, raw)
        digest = hashlib.sha256(raw).hexdigest()
        with self._lock:
            if self._stop.is_set():
                temporary.unlink(missing_ok=True)
                raise ValueError("publisher stopped before publication")
            # One rename and an immutable-state swap. All preparation/fsync
            # stays outside this short critical section used by challenges.
            temporary.replace(self.path)
            self._current, self._authorities = document, authorities
            self._digest = digest
            self._deadline = verified_mono + self.ttl_seconds
        self._sync_directory(self.path)

    def _acknowledge(self, nonce: str) -> bytes:
        with self._lock:
            if (
                self._current is None
                or self._stop.is_set()
                or time.monotonic() >= self._deadline
                or time.time() >= self._current["expires_at"]
                or len(nonce) != 64
                or any(c not in "0123456789abcdef" for c in nonce)
            ):
                return b"{}\n"
            return (
                self._bytes(
                    {
                        "nonce": nonce,
                        "sha256": self._digest,
                    }
                )
                + b"\n"
            )

    def start(self) -> None:
        self._acquire_writer()
        try:
            self.refresh()  # verify fresh; never restore journal grants
            path = Path(str(self.path) + ".sock")
            if path.exists():
                if not stat.S_ISSOCK(path.lstat().st_mode):
                    raise ValueError("publisher socket path is occupied")
                path.unlink()  # exclusive writer lock proves no live publisher owns it
            publisher = self

            class Handler(socketserver.StreamRequestHandler):
                def handle(self) -> None:
                    self.request.settimeout(1)
                    try:
                        nonce = self.rfile.readline(66).decode().strip()
                        self.wfile.write(publisher._acknowledge(nonce))
                    except (OSError, UnicodeError):
                        return

            class Server(socketserver.ThreadingUnixStreamServer):
                daemon_threads = True
                block_on_close = False

            self._server = Server(str(path), Handler)
            path.chmod(0o600)
            self._thread = threading.Thread(
                target=self._run, name="memory-grant-publisher", daemon=True
            )
            self._thread.start()
            self._verification_thread = threading.Thread(
                target=self._verify_loop, name="memory-grant-verifier", daemon=True
            )
            self._verification_thread.start()
        except BaseException:
            self.close()
            raise

    def _run(self) -> None:
        assert self._server is not None
        self._server.serve_forever(poll_interval=0.1)

    def _verify_loop(self) -> None:
        try:
            while not self._stop.wait(self.poll_seconds):
                self.refresh()
        finally:
            if self._stop.is_set():
                self._release_writer()

    @property
    def running(self) -> bool:
        return (
            not self._stop.is_set()
            and self._thread is not None
            and self._thread.is_alive()
            and self._verification_thread is not None
            and self._verification_thread.is_alive()
        )

    def _release_writer(self) -> None:
        if not self._refresh_lock.acquire(timeout=0.1):
            return
        try:
            if self._writer_fd is not None:
                os.close(self._writer_fd)
                self._writer_fd = None
        finally:
            self._refresh_lock.release()

    def close(self) -> None:
        # Bound shutdown even when a host verifier stalls. Do not release the
        # single-writer lock while that thread could still be executing.
        self._stop.set()
        if (
            self._server is not None
            and self._thread is not None
            and self._thread.is_alive()
        ):
            self._server.shutdown()
        if self._thread is not None:
            if self._thread.ident is not None:
                self._thread.join(timeout=2)
            self._thread = None
        if self._server is not None:
            self._server.server_close()
            Path(str(self.path) + ".sock").unlink(missing_ok=True)
            self._server = None
        if self._verification_thread is not None:
            if self._verification_thread.ident is not None:
                self._verification_thread.join(timeout=2)
            if not self._verification_thread.is_alive():
                self._verification_thread = None
        self._release_writer()
