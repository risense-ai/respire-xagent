"""Remote ``MemoryStore`` backed by the memory SaaS HTTP API.

This adapter lets xagent serve persistent memory from the standalone memory
service (``docs/方案-xagent记忆SaaS.md`` §3, §4.5) instead of the local
LanceDB store. It is published in the place ``UserIsolatedMemoryStore``
occupies today, wrapped by ``RevocableMemoryStore`` -- see §3.2.

Division of responsibility (方案 §3.2/§3.4):

* **Ownership and ACL live server-side.** ``add`` never stamps
  ``metadata.user_id``; the service derives the owner from the identity
  headers and overwrites the key. A caller-supplied ``user_id`` is dropped
  from the request body rather than forwarded, so a stale stamp can never trip
  the service's anti-smuggling check.
* **Execution scope stays client-side.** ``execution_scope`` is a cooperative
  namespace, not a security boundary (``core/execution_scope.py``), so the
  stamping, the ``metadata`` filter merge and the
  ``__scope_exclusive__`` directive are reproduced here exactly as
  ``UserIsolatedMemoryStore`` applies them -- the server just executes them.
* **Filters are passed through untouched.** Filter semantics are defined by
  ``MemoryStore._matches_filters`` and must be evaluated server-side (§3.5);
  the client only merges the scope-derived keys it owns.

Failure semantics (§3.5.1/§3.5.2): a transport failure, a timeout, a 5xx or an
authentication rejection raises :class:`RemoteMemoryStoreError` -- an empty
result set must only ever mean "no match". A by-id read/write that is denied
or missing returns the same ``MemoryResponse`` string the local wrapper uses,
so "exists but not visible" is indistinguishable from "does not exist".

Timeouts follow §3.3: reads use a 2s read timeout, writes 5s, over a shared
keep-alive connection pool. The ``MemoryStore`` interface is synchronous and
the web runtime calls it from the event loop; the tight read timeout is the
mitigation for that, not a re-implementation of the routes.

T14 additions (shared space, directed delivery, delegated memory -- §B1):

* ``add`` honours a leading ``[to:<target>]`` marker: that call becomes one
  directed delivery to another user's memory instead of the caller's own note.
* ``update``/``delete`` accept an optional explicit owner, sent as
  ``X-Memory-Owner-Id``, for management and write-on-behalf scenarios.
* ``remote_memory_ready`` probes ``/readyz`` on behalf of the task-startup
  memory policy that ``web/memory_lifecycle.py`` registers.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import threading
import time
from collections.abc import Mapping as AbcMapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Iterator, List, Mapping, Optional, Union
from urllib.parse import quote

import requests

from xagent.config import (
    get_memory_agent_seats_enabled,
    get_memory_backend,
    get_memory_service_key,
    get_memory_service_url,
)
from xagent.core.execution_scope import (
    get_execution_scope,
    memory_dimension_metadata,
    metadata_carries_scope_dimensions,
)
from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.memory.scope_columns import SCOPE_EXCLUSIVE_FILTER_KEY
from xagent.core.user_context import current_user_id

logger = logging.getLogger(__name__)

#: Read path timeout (方案 §3.3): the interactive path must fail fast rather
#: than block the caller's event loop on a slow service.
READ_TIMEOUT_SECONDS = 2.0
#: Write path timeout (方案 §3.3). Longer than the read budget because a write
#: that times out may already have been applied; §3.5.8 also forbids retrying
#: writes, so a slow success must be given room before it is reported failed.
WRITE_TIMEOUT_SECONDS = 5.0
CONNECT_TIMEOUT_SECONDS = 2.0

#: Byte-for-byte the local wrapper's denial message
#: (``web/user_isolated_memory.py``), so callers cannot tell the two paths
#: apart and the service's "exists but not visible" answer does not leak.
NOT_FOUND_OR_DENIED_ERROR = "Memory note not found or access denied"

#: Default reported by ``get_stats`` when the service omits the store type.
DEFAULT_STORE_TYPE = "remote"

#: T14 §B1-1 -- directed delivery. A note whose content starts with
#: ``[to:<target>]`` is delivered to another user's memory through the
#: service's deliver endpoint instead of being written as the caller's own
#: note; the marker's remaining text is the delivered body. A marker with an
#: empty target does not match, so a malformed marker can never silently
#: redirect an ordinary write. The service re-validates the delivery (its own
#: target resolution and category whitelist), so this pattern is a channel,
#: not the authority.
DELIVERY_MARKER_PATTERN = re.compile(r"^\[to:([^\]]+)\]\s*")
DELIVERY_PATH = "/v1/memory/deliver"

#: Judged write path (方案 §10.4). Unlike ``add`` (a plain create), this
#: endpoint runs the four-branch judgement: a high-similarity candidate is
#: merged, a medium one becomes the new note's parent, and anything else is a
#: plain create. The caller reads the chosen branch from the response.
REMEMBER_PATH = "/v1/memory/remember"

#: §A5 to-agent shared-soil marker: ``[share:dept]``, ``[share:dept:<id>]`` or
#: ``[share:company]`` plants the note into the caller's own department (resolved
#: server-side), an explicit department, or the company layer, so an agent can
#: feed its team's common ground without any human opening the console. Same
#: channel-not-authority rule as delivery: the service resolves the space and
#: re-checks write rights; an empty remainder does not match, so a malformed
#: marker can never silently strip an ordinary write.
SHARE_MARKER_PATTERN = re.compile(r"^\[share:(company|dept|department)(?::(\d+))?\]\s*")

#: T14 §B1-2 -- explicit owner header. It addresses *another* user's memory
#: for management and write-on-behalf calls; the service grants it only when
#: the identity's scope covers the target.
OWNER_HEADER = "X-Memory-Owner-Id"

#: T14 §B1-3 -- readiness probe. The answer is cached process-wide for this
#: budget because the probe sits on the task-startup path, where a round trip
#: per task would tax every delegated run.
READY_PATH = "/readyz"
READINESS_CACHE_SECONDS = 30.0
#: Reason reported when the service answered the readiness probe with a
#: failure, a non-2xx status or no answer at all.
READINESS_UNAVAILABLE_REASON = "remote_memory_unavailable"
#: Reason reported when the remote backend is selected without a usable URL.
READINESS_UNCONFIGURED_REASON = "remote_memory_unconfigured"

#: The explicit-owner override of the *current* context (T14 §B1-2). It is a
#: context variable rather than an instance attribute because one adapter
#: instance is shared by every request in the process; an attribute would leak
#: one caller's target owner into a concurrent call.
_explicit_owner_override: ContextVar[Optional[int]] = ContextVar(
    "remote_memory_explicit_owner", default=None
)
_readiness_cache: Optional[tuple[float, bool, Optional[str]]] = None
_readiness_cache_lock = threading.Lock()

#: T14 §0.2 item 6 -- the delegated-agent identity override of the current
#: context. When set, the identity header carries this value (``agent:<ref>``,
#: the seat's ``agent_ref`` on the service) instead of the ambient xagent user
#: id, so a child agent's memory calls land in its own seat while everyone else
#: keeps the historical parent-identity behaviour. A context variable, not an
#: adapter attribute: one store instance is shared by every request in the
#: process, so the switch must travel with the execution context.
_agent_identity_override: ContextVar[Optional[str]] = ContextVar(
    "remote_memory_agent_identity", default=None
)

#: The organization surface answers the seat probe: users filtered by kind.
AGENT_SEATS_PATH = "/v1/org/users"
AGENT_SEATS_PARAMS = {"kind": "agent"}
#: Cached process-wide for this budget, exactly like the readiness probe: the
#: answer sits on the delegation path, where a round trip per delegated run
#: would tax every call.
AGENT_SEATS_CACHE_SECONDS = 30.0
#: Short budget for the probe itself: a slow service must never delay (or fail)
#: a delegation.
AGENT_SEATS_PROBE_TIMEOUT = (1.0, 2.0)
#: Reason reported when the seat probe could not be answered (missing
#: configuration, transport failure, non-2xx or a malformed payload).
AGENT_SEATS_PROBE_FAILED_REASON = "agent_seat_probe_failed"

#: (monotonic time, refs or None, reason or None). ``None`` refs mean the probe
#: failed; the delegation path then keeps the parent identity rather than
#: guessing a seat.
_agent_seats_cache: Optional[tuple[float, Optional[frozenset[str]], Optional[str]]] = (
    None
)
_agent_seats_cache_lock = threading.Lock()

_shared_session: Optional[requests.Session] = None
_shared_session_lock = threading.Lock()


class RemoteMemoryStoreError(RuntimeError):
    """The remote memory service could not complete an operation.

    Raised for transport failures, timeouts, a 5xx, a malformed payload, or a
    rejected credential. ``MemoryStore.search``/``list_all`` document exactly
    this contract: raise, never degrade into an empty result.
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RemoteMemoryStoreAuthenticationError(RemoteMemoryStoreError):
    """The remote memory service rejected the service credentials (HTTP 401)."""


class RemoteMemoryStoreIncompleteError(RemoteMemoryStoreError):
    """A bounded list response cannot satisfy a complete-list request."""


class RemoteMemoryStoreConfigurationError(RuntimeError):
    """The remote backend is selected but not completely configured."""


def _get_shared_session() -> requests.Session:
    """Return the process-wide keep-alive session.

    One session per process gives every adapter instance a shared connection
    pool (§3.3). Credentials are never put on ``session.headers``: they are
    passed per request, so the shared session cannot leak one identity's
    headers into another request.
    """
    global _shared_session
    if _shared_session is None:
        with _shared_session_lock:
            if _shared_session is None:
                session = requests.Session()
                session.headers.update({"User-Agent": "xagent-memory-remote/1"})
                _shared_session = session
    return _shared_session


@contextmanager
def explicit_memory_owner(owner_id: Optional[int]) -> Iterator[None]:
    """Address ``owner_id``'s memory for the duration of the block (T14 §B1-2).

    Sends ``X-Memory-Owner-Id`` on every request made inside the block, which
    the service honours only when the acting identity's scope covers the
    target (``acl``/``notes`` server-side). ``None`` is a no-op: the identity
    header alone decides the owner, exactly as before this hook existed.
    """
    if owner_id is None:
        yield
        return
    token = _explicit_owner_override.set(int(owner_id))
    try:
        yield
    finally:
        _explicit_owner_override.reset(token)


def registered_agent_seats(
    *,
    refresh: bool = False,
    session: Optional[requests.Session] = None,
) -> tuple[Optional[frozenset[str]], Optional[str]]:
    """Return the registered agent seats, or ``(None, reason)`` (T14 §0.2 item 6).

    The xagent adapter asks the memory service which ``agents.id`` values have a
    seat of their own before switching a delegated child's identity; anything
    else would make an unregistered child write under a phantom identity. The
    answer is process-wide cached for ``AGENT_SEATS_CACHE_SECONDS`` -- failures
    included, so a down service is not hammered by every delegation.

    Never raises: "could not tell" must degrade to the historical behaviour
    (the child runs as its parent), never block a delegation.
    """
    global _agent_seats_cache
    now = time.monotonic()
    if not refresh:
        with _agent_seats_cache_lock:
            cached = _agent_seats_cache
        if cached is not None and now - cached[0] < AGENT_SEATS_CACHE_SECONDS:
            return cached[1], cached[2]

    seats, reason = _safe_probe_agent_seats(session)
    with _agent_seats_cache_lock:
        _agent_seats_cache = (now, seats, reason)
    return seats, reason


def _safe_probe_agent_seats(
    session: Optional[requests.Session],
) -> tuple[Optional[frozenset[str]], Optional[str]]:
    """``_probe_agent_seats`` with its "never raises" contract enforced."""
    try:
        return _probe_agent_seats(session)
    except Exception:  # noqa: BLE001 - "could not tell" must degrade, not propagate
        logger.debug("agent seat probe raised; reporting the failure")
        return None, AGENT_SEATS_PROBE_FAILED_REASON


def _probe_agent_seats(
    session: Optional[requests.Session],
) -> tuple[Optional[frozenset[str]], Optional[str]]:
    """One ``GET /v1/org/users?kind=agent``; classify the answer.

    The whole probe runs under a short timeout and swallows every transport or
    payload problem into ``(None, reason)``: an unreachable service, a service
    key without ``org:admin`` (older deployments) or a malformed body all mean
    "no seat information", which the caller treats as "keep the parent identity".
    """
    base_url = get_memory_service_url()
    service_key = get_memory_service_key()
    if not base_url or not service_key:
        return None, READINESS_UNCONFIGURED_REASON
    if session is None:
        session = _get_shared_session()
    url = f"{base_url.rstrip('/')}{AGENT_SEATS_PATH}"
    try:
        response = session.request(
            "GET",
            url,
            params=dict(AGENT_SEATS_PARAMS),
            headers={
                "Authorization": f"Bearer {service_key}",
                "Accept": "application/json",
            },
            timeout=AGENT_SEATS_PROBE_TIMEOUT,
        )
    except requests.RequestException:
        return None, AGENT_SEATS_PROBE_FAILED_REASON
    if response.status_code != 200:
        return None, AGENT_SEATS_PROBE_FAILED_REASON
    try:
        payload = response.json()
    except ValueError:
        return None, AGENT_SEATS_PROBE_FAILED_REASON
    users = payload.get("users") if isinstance(payload, AbcMapping) else None
    if not isinstance(users, list):
        return None, AGENT_SEATS_PROBE_FAILED_REASON
    refs = {
        str(user.get("agent_ref"))
        for user in users
        if isinstance(user, AbcMapping)
        and user.get("kind") == "agent"
        and user.get("agent_ref") is not None
    }
    return frozenset(refs), None


def _unwrap_remote_store(store: Optional[Any]) -> Optional["RemoteMemoryStore"]:
    """Find the ``RemoteMemoryStore`` behind delegation wrappers, if any.

    The delegating task runs on the process-published store -- a chain of
    revocation / user-isolation wrappers over the real adapter -- so a plain
    ``isinstance(store, RemoteMemoryStore)`` would never match in production
    (it only matched in unit tests that passed the bare adapter). Walk the
    wrapper graph by object references (bounded) and return the first remote
    adapter found; ``None`` means "no remote adapter reachable", keeping
    local / in-memory stores on the historical parent identity.
    """
    if store is None:
        return None
    if isinstance(store, RemoteMemoryStore):
        return store
    seen: set[int] = set()
    stack: list[Any] = [store]
    while stack and len(seen) < 16:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, RemoteMemoryStore):
            return current
        if not hasattr(current, "__dict__"):
            continue
        for value in vars(current).values():
            if isinstance(value, MemoryStore) or (
                value is not None and hasattr(value, "search") and hasattr(value, "add")
            ):
                stack.append(value)
    return None


@contextmanager
def delegated_agent_identity(
    agent_id: Optional[int],
    *,
    store: Optional[Any] = None,
    refresh: bool = False,
    session: Optional[requests.Session] = None,
) -> Iterator[None]:
    """Run the block under ``agent_id``'s own seat, when it has one (T14 §0.2).

    The switch engages only when ``store`` reaches the remote adapter -- the
    only backend with an identity to switch, and the only one whose calls reach
    the service that holds the seats -- and the target agent is registered as a
    seat there (``agent_ref`` = the agent id). ``store`` may be the bare
    adapter or the wrapped process-published store; wrappers are unwrapped
    first. Every other case -- the switch disabled, a local or missing store,
    an unregistered agent, a failed probe -- is a no-op, so a delegation is
    never blocked nor pointed at a phantom identity. The override is reset
    even when the block raises.
    """
    if agent_id is None or not get_memory_agent_seats_enabled():
        yield
        return
    if _unwrap_remote_store(store) is None:
        yield
        return
    try:
        ref = str(int(agent_id))
    except (TypeError, ValueError):
        yield
        return
    try:
        seats, _reason = registered_agent_seats(refresh=refresh, session=session)
    except Exception:  # noqa: BLE001 - a probe hiccup must never block a delegation
        logger.debug("agent seat probe raised; keeping the parent identity")
        seats = None
    if not seats or ref not in seats:
        yield
        return
    token = _agent_identity_override.set(f"agent:{ref}")
    try:
        yield
    finally:
        _agent_identity_override.reset(token)


def remote_memory_ready(
    *,
    refresh: bool = False,
    session: Optional[requests.Session] = None,
) -> tuple[bool, Optional[str]]:
    """Probe the memory service's readiness endpoint (T14 §B1-3).

    Returns ``(available, reason)`` with ``reason`` set only when the answer is
    negative. A transport failure, a timeout, a non-2xx status or a missing
    service URL all report unavailable -- never an empty success, so no caller
    can mistake "the service is down" for "memory is fine". The answer is
    cached for ``READINESS_CACHE_SECONDS``; ``refresh=True`` forces a probe.

    Args:
        refresh: Bypass the cached answer.
        session: Optional injected session (tests).
    """
    global _readiness_cache
    now = time.monotonic()
    if not refresh:
        with _readiness_cache_lock:
            cached = _readiness_cache
        if cached is not None and now - cached[0] < READINESS_CACHE_SECONDS:
            return cached[1], cached[2]

    available, reason = _probe_readyz(session)
    with _readiness_cache_lock:
        _readiness_cache = (now, available, reason)
    return available, reason


def _probe_readyz(session: Optional[requests.Session]) -> tuple[bool, Optional[str]]:
    """Send one readiness request and classify the answer."""
    base_url = get_memory_service_url()
    if not base_url:
        return False, READINESS_UNCONFIGURED_REASON
    service_key = get_memory_service_key()
    headers = {"Authorization": f"Bearer {service_key}"} if service_key else {}
    try:
        response = (session if session is not None else _get_shared_session()).get(
            f"{base_url.rstrip('/')}{READY_PATH}",
            headers=headers,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
    except requests.exceptions.RequestException as error:
        logger.warning("Remote memory readiness probe failed: %s", error)
        return False, READINESS_UNAVAILABLE_REASON
    if 200 <= response.status_code < 300:
        return True, None
    logger.warning(
        "Remote memory readiness probe answered HTTP %s", response.status_code
    )
    return False, READINESS_UNAVAILABLE_REASON


def _parse_delivery_marker(content: Any) -> Optional[tuple[str, str]]:
    """Split ``[to:<target>] <body>`` into ``(target, body)`` (T14 §B1-1).

    Returns ``None`` for anything that is not a delivery marker: bytes content,
    a marker that appears later in the text, and a marker whose target is
    empty. The body is the content with the marker and its trailing whitespace
    removed, which is what gets delivered.
    """
    if not isinstance(content, str):
        return None
    match = DELIVERY_MARKER_PATTERN.match(content)
    if match is None:
        return None
    target = match.group(1).strip()
    if not target:
        return None
    return target, content[match.end() :]


def _parse_share_marker(content: Any) -> Optional[tuple[str, Optional[int], str]]:
    """Split ``[share:<kind>[:<dept-id>]] <body>`` into space + stripped body.

    Returns ``None`` for non-marker content, an empty remainder, or bytes.
    ``dept`` and ``department`` both mean the department layer; without an id
    the service resolves the caller's own department (§A5 to-agent default).
    """
    if not isinstance(content, str):
        return None
    match = SHARE_MARKER_PATTERN.match(content)
    if match is None:
        return None
    kind = "company" if match.group(1) == "company" else "department"
    space_id = int(match.group(2)) if match.group(2) else None
    body = content[match.end() :]
    if not body.strip():
        return None
    return kind, space_id, body


def _format_timestamp(value: datetime) -> str:
    """Serialize a note timestamp as RFC3339 without a UTC offset (§3.5.4).

    ``MemoryNote.timestamp`` is naive local time in xagent; an aware value is
    converted to local time and stripped of its offset so the wire format
    stays consistently offset-free.
    """
    if value.tzinfo is not None:
        value = value.astimezone().replace(tzinfo=None)
    return value.isoformat()


def _parse_timestamp(value: Any, *, fallback: datetime) -> datetime:
    """Parse a service timestamp back into naive local time (§3.5.4)."""
    parsed: Optional[datetime] = None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate:
            try:
                parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            except ValueError:
                logger.warning("Ignoring unparsable memory note timestamp")
    if parsed is None:
        return fallback
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _note_from_payload(data: Mapping[str, Any]) -> MemoryNote:
    """Rebuild a ``MemoryNote`` from the service's JSON representation.

    Callers use ``isinstance(x, MemoryNote)`` (``web/api/memory.py``), so the
    adapter must never hand back a raw dict (§3.3.6). Both the ``content`` and
    ``content_b64`` spellings are accepted, as is ``note_ts`` for the
    timestamp column.
    """
    raw_content = data.get("content")
    content: Union[str, bytes]
    if raw_content is None and data.get("content_b64"):
        try:
            content = base64.b64decode(str(data["content_b64"]))
        except (binascii.Error, ValueError):
            logger.warning("Ignoring undecodable content_b64 in memory note payload")
            content = b""
    elif raw_content is None:
        content = ""
    elif isinstance(raw_content, bytes):
        content = raw_content
    else:
        content = str(raw_content)

    raw_metadata = data.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

    note_id = data.get("id") or data.get("memory_id")
    if not note_id:
        raise RemoteMemoryStoreError(
            "Remote memory service returned a note without an id"
        )

    return MemoryNote(
        id=str(note_id),
        content=content,
        keywords=[str(item) for item in (data.get("keywords") or [])],
        tags=[str(item) for item in (data.get("tags") or [])],
        category=str(data.get("category") or "general"),
        timestamp=_parse_timestamp(
            data.get("timestamp", data.get("note_ts")), fallback=datetime.now()
        ),
        mime_type=str(data.get("mime_type") or "text/plain"),
        metadata=metadata,
        space_kind=(str(v) if (v := data.get("space_kind")) else None),
        space_id=(int(v) if (v := data.get("space_id")) is not None else None),
    )


class RemoteMemoryStore(MemoryStore):
    """``MemoryStore`` implementation that talks to the memory SaaS over HTTP.

    All nine abstract methods plus the two scope-maintenance overrides are
    implemented synchronously, matching the ABC (§3.1).
    """

    def __init__(
        self,
        base_url: str,
        service_key: str,
        *,
        session: Optional[requests.Session] = None,
        connect_timeout: float = CONNECT_TIMEOUT_SECONDS,
        read_timeout: float = READ_TIMEOUT_SECONDS,
        write_timeout: float = WRITE_TIMEOUT_SECONDS,
        user_id_provider: Optional[Callable[[], Optional[int]]] = None,
    ) -> None:
        """
        Args:
            base_url: Service base URL, e.g. ``http://memsrv:8080``.
            service_key: Service-level bearer key (never hardcoded here; it
                comes from ``XAGENT_MEMORY_SERVICE_KEY`` via ``config.py``).
            session: Optional injected ``requests.Session`` (tests).
            connect_timeout: TCP connect budget in seconds.
            read_timeout: Read-operation budget in seconds.
            write_timeout: Write-operation budget in seconds.
            user_id_provider: Optional override for the identity contextvar
                lookup (tests).
        """
        if not base_url:
            raise RemoteMemoryStoreConfigurationError("memory service URL is empty")
        if not service_key:
            raise RemoteMemoryStoreConfigurationError("memory service key is empty")
        self._base_url = base_url.rstrip("/")
        self._service_key = service_key
        self._session = session if session is not None else _get_shared_session()
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._user_id_provider = user_id_provider or current_user_id.get

    # ------------------------------------------------------------------
    # identity and transport
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Build request headers (§3.4).

        The service key authenticates the process; ``X-User-External-Id``
        carries the acting xagent user so the service can resolve ownership.
        Inside :func:`delegated_agent_identity` the header carries the seat
        identity (``agent:<ref>``) instead (T14 §0.2 item 6). A missing user id
        is deliberately sent as nothing at all: the service must answer 401
        rather than be handed an anonymous identity.
        """
        headers = {
            "Authorization": f"Bearer {self._service_key}",
            "Content-Type": "application/json",
        }
        agent_identity = _agent_identity_override.get()
        if agent_identity is not None:
            headers["X-User-External-Id"] = agent_identity
        else:
            user_id = self._user_id_provider()
            if user_id is not None:
                headers["X-User-External-Id"] = str(user_id)
        explicit_owner = _explicit_owner_override.get()
        if explicit_owner is not None:
            headers[OWNER_HEADER] = str(explicit_owner)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, str]] = None,
        write: bool = False,
    ) -> requests.Response:
        """Send one request, mapping transport failures to adapter errors.

        Error messages carry the method and path only -- never the response
        body, which could echo memory content or credentials into a log.
        """
        url = f"{self._base_url}{path}"
        read_timeout = self._write_timeout if write else self._read_timeout
        timeout = (self._connect_timeout, read_timeout)
        try:
            return self._session.request(
                method,
                url,
                json=body,
                params=params,
                headers=self._headers(),
                timeout=timeout,
            )
        except requests.exceptions.Timeout as error:
            raise RemoteMemoryStoreError(
                f"Remote memory request timed out: {method} {path}"
            ) from error
        except requests.exceptions.RequestException as error:
            raise RemoteMemoryStoreError(
                f"Remote memory request failed: {method} {path}"
            ) from error

    @staticmethod
    def _json(path: str, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise RemoteMemoryStoreError(
                f"Remote memory service returned a non-JSON response: {path}"
            ) from error

    @staticmethod
    def _raise_for_error(path: str, response: requests.Response) -> None:
        """Raise unless the service answered with a 2xx (§3.5.1)."""
        status = response.status_code
        if 200 <= status < 300:
            return
        # the service's own message carries the repair ("send a follow-up delivery
        # instead" for a 409) — dropping it left the model retrying blind
        detail = ""
        try:
            payload = response.json()
            message = ((payload or {}).get("error") or {}).get("message")
            if isinstance(message, str) and message:
                detail = f" — {message}"
        except ValueError:
            pass
        if status == 401:
            raise RemoteMemoryStoreAuthenticationError(
                f"Remote memory service rejected the service credential: {path}",
                status_code=status,
            )
        raise RemoteMemoryStoreError(
            f"Remote memory service returned HTTP {status}: {path}{detail}",
            status_code=status,
        )

    @staticmethod
    def _note_items(path: str, payload: Any) -> List[Mapping[str, Any]]:
        """Normalize a note-list payload.

        The service may answer with a bare array or wrap it under
        ``results``/``memories``/``notes``; anything else is a protocol
        violation and must raise rather than be read as "no matches".
        """
        if isinstance(payload, list):
            items: Any = payload
        elif isinstance(payload, dict):
            items = None
            for key in ("results", "memories", "notes"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    items = candidate
                    break
            if items is None:
                raise RemoteMemoryStoreError(
                    f"Remote memory service returned an unexpected payload: {path}"
                )
        else:
            raise RemoteMemoryStoreError(
                f"Remote memory service returned an unexpected payload: {path}"
            )
        for item in items:
            if not isinstance(item, AbcMapping):
                raise RemoteMemoryStoreError(
                    f"Remote memory service returned a non-object note: {path}"
                )
        return list(items)

    # ------------------------------------------------------------------
    # scope helpers (mirrors web/user_isolated_memory.py, minus user_id)
    # ------------------------------------------------------------------

    def _filters_payload(
        self, filters: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Attach this client's scope filters and pass everything else through.

        Without an active scope the caller's filters are forwarded verbatim
        (shallow copy, so a caller-reused dict is never mutated). With a scope,
        the scope's dimension stamps are merged into the ``metadata`` filter
        and ``strict_memory_isolation`` adds the ``__scope_exclusive__``
        directive -- identical to ``UserIsolatedMemoryStore._add_user_filter``,
        except that no ``user_id`` filter is ever added: ownership is the
        service's job (§3.2).
        """
        scope = get_execution_scope()
        dimension_metadata = memory_dimension_metadata(scope)
        exclusive = bool(
            scope is not None
            and scope.strict_memory_isolation
            and not scope.memory_dimensions
        )
        if not dimension_metadata and not exclusive:
            return dict(filters) if filters else None

        merged = dict(filters) if filters else {}
        if dimension_metadata:
            raw_metadata = merged.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            metadata.update(dimension_metadata)
            merged["metadata"] = metadata
        if exclusive:
            merged[SCOPE_EXCLUSIVE_FILTER_KEY] = True
        return merged

    @staticmethod
    def _scope_gates_by_id_access() -> bool:
        """Whether the active scope constrains by-id (get/update/delete)."""
        scope = get_execution_scope()
        if scope is None:
            return False
        return bool(scope.memory_dimensions) or scope.strict_memory_isolation

    @staticmethod
    def _scope_permits(note: MemoryNote) -> bool:
        """Whether the active scope may touch ``note`` by id.

        Mirrors the filter semantics of ``search``/``list_all``; scope is a
        cooperative namespace, so this is the only place by-id access is
        narrowed on the client. Ownership is enforced by the service.
        """
        scope = get_execution_scope()
        dimension_metadata = memory_dimension_metadata(scope)
        if dimension_metadata:
            return all(
                note.metadata.get(key) == value
                for key, value in dimension_metadata.items()
            )
        if scope is not None and scope.strict_memory_isolation:
            return not metadata_carries_scope_dimensions(note.metadata)
        return True

    @staticmethod
    def _not_found_or_denied(note_id: str) -> MemoryResponse:
        return MemoryResponse(
            success=False,
            error=NOT_FOUND_OR_DENIED_ERROR,
            memory_id=note_id,
        )

    def _note_payload(
        self, note: MemoryNote, content_text: Optional[str] = None
    ) -> dict[str, Any]:
        """Build the wire representation of a note.

        A caller-supplied ``user_id`` is dropped: the service owns ownership
        and rejects a smuggled mismatch (§3.4.3), so forwarding a stale value
        could only turn a valid write into a 409.

        ``content_text`` overrides the serialized content (marker-stripped body
        from a ``[share:…]`` prefix); it is only used for str content.
        """
        metadata = dict(note.metadata)
        metadata.pop("user_id", None)
        payload: dict[str, Any] = {
            "id": note.id,
            "keywords": list(note.keywords),
            "tags": list(note.tags),
            "category": note.category,
            "timestamp": _format_timestamp(note.timestamp),
            "mime_type": note.mime_type,
            "metadata": metadata,
        }
        if isinstance(note.content, bytes):
            payload["content_b64"] = base64.b64encode(note.content).decode("ascii")
        else:
            payload["content"] = (
                content_text if content_text is not None else str(note.content)
            )
        return payload

    @staticmethod
    def _note_path(note_id: str) -> str:
        return f"/v1/memory/notes/{quote(str(note_id), safe='')}"

    # ------------------------------------------------------------------
    # MemoryStore surface
    # ------------------------------------------------------------------

    def add(self, note: MemoryNote) -> MemoryResponse:
        # Stamp the active scope's dimensions so scoped searches can filter on
        # them; the owner is set by the service, not stamped here (§3.2).
        note.metadata.update(memory_dimension_metadata(get_execution_scope()))
        delivery = _parse_delivery_marker(note.content)
        if delivery is not None:
            return self._deliver(delivery[0], delivery[1], note)
        share = _parse_share_marker(note.content)
        if share is not None:
            space_kind, space_id, stripped = share
            payload = self._note_payload(note, content_text=stripped)
            payload["space_kind"] = space_kind
            if space_id is not None:
                payload["space_id"] = space_id
        else:
            payload = self._note_payload(note)
        path = "/v1/memory/notes"
        response = self._request("POST", path, body=payload, write=True)
        self._raise_for_error(path, response)
        return MemoryResponse(
            success=True,
            memory_id=self._response_memory_id(response, note.id),
            content=None,
        )

    def _deliver(self, target: str, content: str, note: MemoryNote) -> MemoryResponse:
        """Deliver one note to ``target`` instead of writing it (T14 §B1-1).

        The body follows the service's deliver contract: the target, the
        stripped content, the note's category and its metadata (never a
        ``user_id`` -- ownership is the service's). The keyword and tag lists
        ride along only when the caller set them. Failures keep the existing
        exception semantics (§3.5.1): a transport failure, a rejection (an
        unknown target, a category the service refuses) or a rate limit raises,
        so a delivery that did not happen is never reported as a written note.
        """
        metadata = dict(note.metadata)
        metadata.pop("user_id", None)
        body: dict[str, Any] = {
            "target": target,
            "content": content,
            "category": note.category,
            "metadata": metadata,
        }
        if note.keywords:
            body["keywords"] = list(note.keywords)
        if note.tags:
            body["tags"] = list(note.tags)
        response = self._request("POST", DELIVERY_PATH, body=body, write=True)
        self._raise_for_error(DELIVERY_PATH, response)
        return MemoryResponse(
            success=True,
            memory_id=self._response_memory_id(response, note.id),
            content=None,
        )

    @staticmethod
    def _response_memory_id(response: requests.Response, fallback: str) -> str:
        """Read the service-assigned note id, tolerating a missing body."""
        try:
            payload: Any = response.json()
        except ValueError:
            # A 2xx with an empty body is acceptable: the id is client-side.
            return fallback
        if isinstance(payload, dict):
            return str(payload.get("id") or payload.get("memory_id") or fallback)
        return fallback

    def get(self, note_id: str) -> MemoryResponse:
        path = self._note_path(note_id)
        response = self._request("GET", path)
        if response.status_code in (403, 404):
            return self._not_found_or_denied(note_id)
        self._raise_for_error(path, response)

        payload = self._json(path, response)
        if isinstance(payload, dict) and isinstance(payload.get("note"), dict):
            payload = payload["note"]
        if not isinstance(payload, AbcMapping):
            raise RemoteMemoryStoreError(
                f"Remote memory service returned an unexpected note: {path}"
            )
        note = _note_from_payload(payload)
        if not self._scope_permits(note):
            return self._not_found_or_denied(note_id)
        return MemoryResponse(success=True, memory_id=note.id, content=note)

    def update(
        self, note: MemoryNote, *, owner_id: Optional[int] = None
    ) -> MemoryResponse:
        """Update a note, optionally in ``owner_id``'s memory (T14 §B1-2).

        ``owner_id`` sends ``X-Memory-Owner-Id`` for this call only; the
        service answers 403 unless the acting identity may write that user's
        memory, which surfaces here as the usual denial message. The base
        ``MemoryStore`` signature is unchanged -- the parameter is keyword-only
        and optional.
        """
        with explicit_memory_owner(owner_id):
            return self._update(note)

    def _update(self, note: MemoryNote) -> MemoryResponse:
        # Verify scope access first; an inaccessible note must not be written
        # even though the service's owner check would also refuse it.
        if note.id and self._scope_gates_by_id_access():
            existing = self.get(note.id)
            if not existing.success:
                return existing

        path = self._note_path(note.id)
        response = self._request("PUT", path, body=self._note_payload(note), write=True)
        if response.status_code in (403, 404):
            return self._not_found_or_denied(note.id)
        self._raise_for_error(path, response)
        return MemoryResponse(success=True, memory_id=note.id)

    def delete(self, note_id: str, *, owner_id: Optional[int] = None) -> MemoryResponse:
        """Delete a note, optionally in ``owner_id``'s memory (T14 §B1-2).

        Same explicit-owner contract as :meth:`update`.
        """
        with explicit_memory_owner(owner_id):
            return self._delete(note_id)

    def _delete(self, note_id: str) -> MemoryResponse:
        if self._scope_gates_by_id_access():
            existing = self.get(note_id)
            if not existing.success:
                return existing

        path = self._note_path(note_id)
        response = self._request("DELETE", path, write=True)
        if response.status_code in (403, 404):
            return self._not_found_or_denied(note_id)
        self._raise_for_error(path, response)
        return MemoryResponse(success=True, memory_id=note_id)

    def remember(self, note: MemoryNote, *, force: bool = False) -> MemoryResponse:
        """Write ``note`` through the service's four-branch judgement (§10.4).

        Unlike :meth:`add` this is not a plain create: the service compares the
        body against the owner's earlier notes and picks a branch from the
        cosine similarity of the best candidate -- at or above the merge
        threshold the new text is merged into that note, at or above the parent
        threshold the new note is attached under the matched one, and anything
        else is created (the same three thresholds ``remember`` documents:
        0.50 / 0.35, tunable in the service's ``settings`` row).

        ``force`` skips the judgement and always creates.

        The branch decision comes back in ``MemoryResponse.metadata`` as
        ``action`` (``merge`` / ``parent`` / ``create``) plus ``parent_id``,
        so the caller can report what happened instead of guessing. Failures
        keep the adapter's usual semantics (§3.5.1): transport errors and
        rejections raise, so a write that did not happen is never reported as
        stored.
        """
        # Same scope stamping as ``add``: the dimensions stay a client-side
        # namespace (方案 §3.2) so scoped searches keep matching the notes this
        # path writes; ownership remains the service's job.
        note.metadata.update(memory_dimension_metadata(get_execution_scope()))
        metadata = dict(note.metadata)
        metadata.pop("user_id", None)
        body: dict[str, Any] = {
            "content": str(note.content),
            "category": note.category,
            "keywords": list(note.keywords),
            "tags": list(note.tags),
            "timestamp": _format_timestamp(note.timestamp),
            "mime_type": note.mime_type,
            "metadata": metadata,
            "force": bool(force),
        }
        kind = metadata.get("kind")
        if isinstance(kind, str) and kind:
            body["kind"] = kind
        response = self._request("POST", REMEMBER_PATH, body=body, write=True)
        self._raise_for_error(REMEMBER_PATH, response)
        payload = self._json(REMEMBER_PATH, response)
        if not isinstance(payload, dict):
            raise RemoteMemoryStoreError(
                f"Remote memory service returned an unexpected payload: {REMEMBER_PATH}"
            )
        action = str(payload.get("action") or "create")
        return MemoryResponse(
            success=True,
            memory_id=str(payload.get("id") or note.id),
            metadata={
                "action": action,
                "parent_id": payload.get("parent_id"),
            },
        )

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[MemoryNote]:
        path = "/v1/memory/search"
        body = {
            "query": query,
            "k": k,
            "filters": self._filters_payload(filters),
            "similarity_threshold": similarity_threshold,
        }
        response = self._request("POST", path, body=body)
        self._raise_for_error(path, response)
        items = self._note_items(path, self._json(path, response))
        return [_note_from_payload(item) for item in items]

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        path = "/v1/memory/list"
        response = self._request(
            "POST", path, body={"filters": self._filters_payload(filters)}
        )
        self._raise_for_error(path, response)
        payload = self._json(path, response)
        # Completeness belongs to this endpoint's envelope. Search and inbox
        # intentionally use bounded results and keep their extraction contract.
        if not isinstance(payload, dict) or type(payload.get("truncated")) is not bool:
            raise RemoteMemoryStoreError(
                "Remote memory list response has no valid completeness flag"
            )
        if payload["truncated"]:
            raise RemoteMemoryStoreIncompleteError(
                "Remote memory list is incomplete; narrow the filters before retrying"
            )
        items = self._note_items(path, payload)
        return [_note_from_payload(item) for item in items]

    def inbox_list(self, include_read: bool = False) -> List[MemoryNote]:
        """Deliveries addressed to the identity; unread-only unless ``include_read``.

        Read state lives beside the immutable delivery row (service migration 0009), so
        the tool can distinguish "new" from "already handled" without mutating anything.
        """
        path = "/v1/memory/inbox/list"
        response = self._request(
            "POST", path, body={"include_read": bool(include_read)}
        )
        self._raise_for_error(path, response)
        items = self._note_items(path, self._json(path, response))
        return [_note_from_payload(item) for item in items]

    def inbox_read(self, ids: List[str]) -> int:
        """Mark deliveries as read; returns how many receipts were written."""
        path = "/v1/memory/inbox/read"
        response = self._request("POST", path, body={"ids": [str(i) for i in ids]})
        self._raise_for_error(path, response)
        payload = self._json(path, response)
        try:
            return int(payload.get("marked") or 0)
        except (AttributeError, TypeError, ValueError):
            return 0

    def clear(self) -> None:
        path = "/v1/memory/clear"
        # The service clears only the identity user's notes (§3.5.5).
        response = self._request("POST", path, write=True)
        self._raise_for_error(path, response)

    def get_stats(self) -> dict[str, Any]:
        path = "/v1/memory/stats"
        response = self._request("GET", path)
        self._raise_for_error(path, response)
        payload = self._json(path, response)
        if not isinstance(payload, dict):
            raise RemoteMemoryStoreError(
                f"Remote memory service returned a non-object stats payload: {path}"
            )
        stats = dict(payload)
        category_counts = stats.get("category_counts")
        if "total_count" not in stats:
            total = 0
            if isinstance(category_counts, dict):
                total = sum(
                    value
                    for value in category_counts.values()
                    if isinstance(value, int)
                )
            stats["total_count"] = total
        if not isinstance(category_counts, dict):
            stats["category_counts"] = {}
        if not isinstance(stats.get("tag_counts"), dict):
            stats["tag_counts"] = {}
        stats["memory_store_type"] = str(
            stats.get("memory_store_type") or DEFAULT_STORE_TYPE
        )
        return stats

    def delete_by_scope_dimension(self, dim_key: str, value: Any) -> MemoryResponse:
        """Bulk-delete notes stamped ``dim_key=value`` via the service endpoint.

        Maintenance-only: the service requires a ``scope:maintenance`` key and
        answers 403 otherwise, which surfaces here as
        :class:`RemoteMemoryStoreError` rather than a silent no-op.
        """
        path = "/v1/memory/scope/delete-by-dimension"
        response = self._request(
            "POST", path, body={"dim_key": dim_key, "value": str(value)}, write=True
        )
        self._raise_for_error(path, response)
        payload = self._json(path, response)
        deleted = payload.get("deleted_count") if isinstance(payload, dict) else 0
        if not isinstance(deleted, int):
            deleted = 0
        return MemoryResponse(success=True, metadata={"deleted_count": deleted})

    def list_scope_dimension_values(self, dim_key: str) -> set[str]:
        """Distinct stamped values for one scope dimension, service-wide.

        Raises on failure, per the base contract, so a reconciler never
        mistakes an error for "no values".
        """
        path = "/v1/memory/scope/values"
        response = self._request("GET", path, params={"dim": dim_key})
        self._raise_for_error(path, response)
        payload = self._json(path, response)
        values: Any = payload
        if isinstance(payload, dict):
            values = payload.get("values")
        if not isinstance(values, list):
            raise RemoteMemoryStoreError(
                f"Remote memory service returned an unexpected scope payload: {path}"
            )
        return {str(value) for value in values}


def remote_memory_store_from_config() -> Optional[RemoteMemoryStore]:
    """Build the remote store when the backend switch selects it.

    Returns ``None`` when ``XAGENT_MEMORY_BACKEND`` is not ``remote``, so the
    caller keeps its existing admission path. Raises
    :class:`RemoteMemoryStoreConfigurationError` when the switch selects a
    remote backend that is missing its URL or key -- never falls back to the
    local store silently.

    The key is read from configuration only; it is never defaulted, logged, or
    inlined here (§3.4.2).
    """
    if get_memory_backend() != "remote":
        return None
    base_url = get_memory_service_url()
    service_key = get_memory_service_key()
    if not base_url or not service_key:
        raise RemoteMemoryStoreConfigurationError(
            "XAGENT_MEMORY_BACKEND=remote requires XAGENT_MEMORY_SERVICE_URL "
            "and XAGENT_MEMORY_SERVICE_KEY"
        )
    return RemoteMemoryStore(base_url, service_key)


__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "DELIVERY_PATH",
    "NOT_FOUND_OR_DENIED_ERROR",
    "OWNER_HEADER",
    "READINESS_CACHE_SECONDS",
    "READINESS_UNAVAILABLE_REASON",
    "READINESS_UNCONFIGURED_REASON",
    "READY_PATH",
    "READ_TIMEOUT_SECONDS",
    "REMEMBER_PATH",
    "RemoteMemoryStore",
    "RemoteMemoryStoreAuthenticationError",
    "RemoteMemoryStoreConfigurationError",
    "RemoteMemoryStoreError",
    "RemoteMemoryStoreIncompleteError",
    "WRITE_TIMEOUT_SECONDS",
    "explicit_memory_owner",
    "remote_memory_ready",
    "remote_memory_store_from_config",
]
