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
import hashlib
import logging
import re
import threading
import time
from collections.abc import Mapping as AbcMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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

from .agent_bindings import ApprovedAgentBinding, approved_bindings, find_binding
from .agent_seats import (
    AgentSeatRequest,
    AgentSeatSnapshot,
    read_agent_seat_snapshot,
)

logger = logging.getLogger(__name__)

# Client-only opt-in key: pass ``filters={SCOPE_DIMENSIONS_FILTER_KEY: True}``
# to merge the active scope's memory dimensions into a search/list filter.
# Dimensions are otherwise stamp-only (cooperative tags), so a task-scoped
# writer never blinds cross-task recall by default. The key is stripped
# before the request reaches the service.
SCOPE_DIMENSIONS_FILTER_KEY = "__scope_dimensions__"

#: Read path timeout (方案 §3.3): the interactive path must fail fast rather
#: than block the caller's event loop on a slow service.
READ_TIMEOUT_SECONDS = 2.0
#: Write path timeout (方案 §3.3). Longer than the read budget because a write
#: that times out may already have been applied; §3.5.8 also forbids retrying
#: writes, so a slow success must be given room before it is reported failed.
WRITE_TIMEOUT_SECONDS = 5.0
CONNECT_TIMEOUT_SECONDS = 2.0

#: User-Agent sent on every outbound call to the memory service. One shared
#: value on the session and on each per-request header dict; supersedes the
#: previous "xagent-memory-remote/1" that was set on the session alone.
MEMORY_SERVICE_USER_AGENT = "xagent-memsrv-client/1.0"

#: Byte-for-byte the local wrapper's denial message
#: (``web/user_isolated_memory.py``), so callers cannot tell the two paths
#: apart and the service's "exists but not visible" answer does not leak.
NOT_FOUND_OR_DENIED_ERROR = "Memory note not found or access denied"

#: Default reported by ``get_stats`` when the service omits the store type.
DEFAULT_STORE_TYPE = "remote"

#: Cosine-**distance** ceiling sent to ``/v1/memory/search`` when the caller
#: does not pass one. The service's ``similarity_threshold`` is a distance
#: ceiling (smaller = stricter) while the ``score`` it returns is similarity
#: (larger = closer), so 0.40 means "similarity >= 0.60" -- not the 1.5 that
#: was used as a placeholder and filtered nothing.
#:
#: Measured on 2026-10-29 against the ext2 stock: relevant notes scored
#: 0.7585 ("年度报销额度", distance 0.2415) and 0.7510 ("供应商对账口径",
#: 0.2490); unrelated notes sat at 0.42-0.52 similarity (0.48-0.58 distance);
#: the nonsense query "zzz不存在的词" topped out at 0.4576 (0.5424). 0.40
#: separates the relevant from the irrelevant; anything near 0.70 lets most of
#: the store back in.
DEFAULT_SEARCH_SIMILARITY_THRESHOLD = 0.40

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
READINESS_DEGRADED_REASON = "remote_memory_degraded"
READINESS_PROTOCOL_ERROR_REASON = "remote_memory_protocol_error"

#: Identity-surface probe used only to *classify* a 401 (never on the happy
#: path). The service answers this with a bare key (no identity header), so a
#: 200 here proves the key is valid and the caller's 401 came from the
#: unrecognised ``X-User-External-Id`` instead.
WHOAMI_PATH = "/v1/whoami"
#: Per-key cache for that classification probe. Bounded so a key rotation is
#: noticed quickly, and keyed by the key fingerprint so concurrent identities
#: never share a verdict.
KEY_VALIDITY_CACHE_SECONDS = 60.0
_key_validity_cache: dict[str, tuple[float, Optional[bool]]] = {}
_key_validity_cache_lock = threading.Lock()

#: Upstream health signal of this process (缺陷二/store-info). Every answered
#: HTTP response -- even a 4xx/5xx -- proves the upstream answered; only a
#: transport failure records "unreachable". ``get_store_info`` folds this into
#: its state instead of reporting the local lifecycle state as if it described
#: the upstream.
_upstream_health: dict[str, Any] = {
    "last_success_at": None,  # wall-clock epoch of the last answered call
    "last_success_status": None,
    "last_failure_at": None,  # wall-clock epoch of the last transport failure
    "last_failure_reason": None,
    "last_event_at": None,  # epoch of whichever event came last
    "last_event_kind": None,  # "answer" | "transport_error"
    "last_status": None,  # status of the last answered call
    "last_path": None,
    "last_error_code": None,
    "last_error_message": None,
    "consecutive_transport_failures": 0,
}
_upstream_health_lock = threading.Lock()


def _record_upstream_answer(path: str, response: requests.Response) -> None:
    """Record that the upstream answered (any HTTP status counts as served)."""
    payload: Any
    try:
        payload = response.json()
    except (ValueError, RecursionError):
        payload = None
    error = ((payload or {}).get("error") or {}) if isinstance(payload, dict) else {}
    error = error if isinstance(error, dict) else {}
    now = time.time()
    with _upstream_health_lock:
        _upstream_health["last_event_at"] = now
        _upstream_health["last_event_kind"] = "answer"
        _upstream_health["last_status"] = response.status_code
        _upstream_health["last_path"] = path
        _upstream_health["last_error_code"] = error.get("code")
        _upstream_health["last_error_message"] = error.get("message")
        _upstream_health["consecutive_transport_failures"] = 0
        if 200 <= response.status_code < 300:
            _upstream_health["last_success_at"] = now
            _upstream_health["last_success_status"] = response.status_code


def _record_transport_failure(path: str, reason: str) -> None:
    """Record that the upstream could not be reached at all."""
    now = time.time()
    with _upstream_health_lock:
        _upstream_health["last_event_at"] = now
        _upstream_health["last_event_kind"] = "transport_error"
        _upstream_health["last_failure_at"] = now
        _upstream_health["last_failure_reason"] = reason
        _upstream_health["last_path"] = path
        _upstream_health["consecutive_transport_failures"] += 1


def remote_upstream_health() -> dict[str, Any]:
    """Snapshot of the upstream health signal (copy; thread-safe)."""
    with _upstream_health_lock:
        return dict(_upstream_health)


def _format_event_time(epoch: Optional[float]) -> str:
    if epoch is None:
        return "从未"
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")


def remote_upstream_report() -> tuple[str, str]:
    """Honest ``(state, detail)`` for the upstream behind a remote store.

    ``state`` is one of ``ready`` / ``degraded`` / ``unavailable``:

    * ``ready`` -- the latest signal is an answered call below 500, or (with
      no call yet this process) a passing ``/readyz`` probe;
    * ``degraded`` -- the latest signal is a 5xx answer, or a transport
      failure after the upstream had answered successfully before;
    * ``unavailable`` -- the upstream has never answered in this process and
      the latest signal is a failure.

    The ``/readyz`` fallback is the probe ``remote_memory_ready`` already
    caches (``READINESS_CACHE_SECONDS``), so this costs no extra traffic on
    repeated reads and never probes per memory request.
    """
    health = remote_upstream_health()
    event_kind = health["last_event_kind"]
    if event_kind is None:
        available, reason = remote_memory_ready()
        now = _format_event_time(time.time())
        if available:
            return "ready", f"本进程尚无上游调用记录；/readyz 探测可达（{now}）"
        return (
            "unavailable",
            f"本进程尚无上游调用记录；/readyz 探测不可达（reason={reason}，{now}）",
        )
    if event_kind == "transport_error":
        detail = (
            f"上游最近一次调用不可达（{health['last_failure_reason']}）："
            f"{health['last_path']}，连续失败 "
            f"{health['consecutive_transport_failures']} 次，"
            f"{_format_event_time(health['last_event_at'])}"
        )
        if health["last_success_at"] is None:
            return "unavailable", detail
        return "degraded", detail
    status = health["last_status"]
    when = _format_event_time(health["last_event_at"])
    if status is not None and status >= 500:
        code = health["last_error_code"]
        code_note = f"（code={code}）" if code else ""
        return (
            "degraded",
            f"上游最近一次调用返回 HTTP {status}{code_note}："
            f"{health['last_path']}，{when}",
        )
    return (
        "ready",
        f"上游最近一次调用成功（HTTP {status}）：{health['last_path']}，{when}",
    )


IDENTITY_REQUIRED_REASON = "remote_memory_identity_required"
AGENT_SEAT_REQUIRED_REASON = "remote_memory_agent_seat_required"

#: The explicit-owner override of the *current* context (T14 §B1-2). It is a
#: context variable rather than an instance attribute because one adapter
#: instance is shared by every request in the process; an attribute would leak
#: one caller's target owner into a concurrent call.
_explicit_owner_override: ContextVar[Optional[int]] = ContextVar(
    "remote_memory_explicit_owner", default=None
)
_readiness_cache: tuple[float, tuple[str | None, str], RemoteMemoryReadiness] | None = (
    None
)
_readiness_cache_lock = threading.Lock()


@dataclass(frozen=True)
class _ParentIdentityBinding:
    user_external_id: str
    service_url: str
    credential_fingerprint: str
    invalidated: threading.Event = field(
        default_factory=threading.Event, kw_only=True, repr=False, compare=False
    )


@dataclass(frozen=True)
class _AgentIdentityBinding(_ParentIdentityBinding):
    snapshot: AgentSeatSnapshot
    provider_generation: int
    agent_ref: str
    owner_check: Callable[[], None] | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class _ConfiguredAgentIdentityBinding(_ParentIdentityBinding):
    mapping: ApprovedAgentBinding
    agent_ref: str
    owner_check: Callable[[], None] | None = field(default=None, kw_only=True)


_AGENT_BINDING_TYPES = (_AgentIdentityBinding, _ConfiguredAgentIdentityBinding)


_agent_identity_override: ContextVar[_ParentIdentityBinding | None] = ContextVar(
    "remote_memory_agent_identity", default=None
)
AGENT_SEATS_PROBE_FAILED_REASON = "agent_seat_probe_failed"

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


class RemoteMemoryStoreIdentityError(RemoteMemoryStoreError):
    """No trusted user identity or provisioned independent seat is available."""


class RemoteMemoryStoreConfigurationError(RuntimeError):
    """The remote backend is selected but not completely configured."""


_USE_CONTEXT_IDENTITY = object()


def require_remote_memory_identity(user_id: Any = _USE_CONTEXT_IDENTITY) -> str:
    """Use only the host's authenticated User.id context, preserving its mapping.

    User context is established by JWT/DB authentication or the trusted runtime,
    never by copying incoming external-id or owner headers. No I/O is allowed
    before this guard on a user-operation path.
    """
    value = current_user_id.get() if user_id is _USE_CONTEXT_IDENTITY else user_id
    if type(value) is not int or value <= 0:
        raise RemoteMemoryStoreIdentityError(IDENTITY_REQUIRED_REASON)
    return str(value)


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
                session.headers.update({"User-Agent": MEMORY_SERVICE_USER_AGENT})
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


def _configured_identity_target() -> tuple[str, str]:
    url, key = get_memory_service_url(), get_memory_service_key()
    if not url or not key:
        raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON)
    return url.rstrip("/"), hashlib.sha256(key.encode()).hexdigest()


def _seat_snapshot(identity: str) -> tuple[AgentSeatSnapshot, int]:
    service_url, profile = _configured_identity_target()
    try:
        return read_agent_seat_snapshot(
            AgentSeatRequest(service_url, identity, profile)
        )
    except Exception:
        # Provider errors must not expose control-plane data or trigger I/O.
        raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON) from None


def registered_agent_seats(
    *,
    user_id: Any = _USE_CONTEXT_IDENTITY,
    refresh: bool = False,
    session: requests.Session | None = None,
    cached_only: bool = True,
) -> tuple[frozenset[str] | None, str | None]:
    """Read approved initialization records or a trusted host mapping, without I/O.

    Legacy refresh/session/cached_only arguments are accepted for compatibility
    but cannot refresh a mapping or trigger I/O. The host must publish/refresh
    through its trusted provider with separate control-plane credentials.
    """
    identity = require_remote_memory_identity(user_id)
    try:
        mappings = approved_bindings()
        if mappings is not None:
            url, profile = _configured_identity_target()
            refs = frozenset(
                str(b.agent_id)
                for b in mappings
                if str(b.user_id) == identity
                and b.service_url == url
                and b.runtime_profile == profile
            )
            return (refs, None) if refs else (None, AGENT_SEAT_REQUIRED_REASON)
        snapshot, _ = _seat_snapshot(identity)
        return snapshot.agent_refs, None
    except (
        RemoteMemoryStoreIdentityError,
        ValueError,
        TypeError,
        RecursionError,
        OverflowError,
    ):
        return None, AGENT_SEAT_REQUIRED_REASON


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


def _delegated_identity_binding(
    agent_id: int | None,
    *,
    user_id: Any = _USE_CONTEXT_IDENTITY,
    owner_check: Callable[[], None] | None = None,
) -> _ParentIdentityBinding:
    identity = require_remote_memory_identity(user_id)
    url, fingerprint = _configured_identity_target()
    if not get_memory_agent_seats_enabled():
        return _ParentIdentityBinding(identity, url, fingerprint)
    if type(agent_id) is not int or agent_id <= 0:
        raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON)
    try:
        mapping = find_binding(identity, agent_id, url, fingerprint)
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON) from None
    if mapping is not None:
        return _ConfiguredAgentIdentityBinding(
            identity, url, fingerprint, mapping, str(agent_id), owner_check=owner_check
        )
    snapshot, epoch = _seat_snapshot(identity)
    ref = str(agent_id)
    if ref not in snapshot.agent_refs:
        raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON)
    return _AgentIdentityBinding(
        identity, url, fingerprint, snapshot, epoch, ref, owner_check=owner_check
    )


def require_delegated_memory_identity(
    agent_id: int | None,
    *,
    user_id: Any = _USE_CONTEXT_IDENTITY,
) -> str:
    """Resolve a known identity locally before any readiness or data request."""
    binding = _delegated_identity_binding(agent_id, user_id=user_id)
    return (
        f"agent:{binding.agent_ref}"
        if isinstance(binding, _AGENT_BINDING_TYPES)
        else binding.user_external_id
    )


@contextmanager
def delegated_agent_identity(
    agent_id: int | None,
    *,
    store: Any | None = None,
    refresh: bool = False,
    session: requests.Session | None = None,
    owner_check: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Bind entry identity/config and a trusted mapping, without parent fallback.

    Approved initialization records require a backend ownership check and use
    existing whoami to verify the selected IDs at entry. Legacy host providers
    remain compatible. refresh/session never trigger organization enumeration.
    """
    remote_store = _unwrap_remote_store(store)
    if remote_store is None:
        yield
        return
    binding = _delegated_identity_binding(
        agent_id, user_id=remote_store._user_id_provider(), owner_check=owner_check
    )
    remote_store._validate_identity_binding(binding)
    if isinstance(binding, _ConfiguredAgentIdentityBinding):
        remote_store._check_configured_mapping_selection(binding)
    token = _agent_identity_override.set(binding)
    try:
        yield
    finally:
        _agent_identity_override.reset(token)


@dataclass(frozen=True)
class RemoteMemoryReadiness:
    """Dependency health only; never evidence of a user's authorization.

    Semantic capability describes the current embedder's state, not historical
    vector coverage or reindex completion. Snapshots may be cached for 30s and
    therefore do not promise real-time availability.
    """

    status: str
    semantic_available: bool = False
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status in {"ready", "degraded"}


def remote_memory_readiness(
    *,
    refresh: bool = False,
    session: Optional[requests.Session] = None,
) -> RemoteMemoryReadiness:
    """Return health and semantic capability, cached by URL and credential.

    This is a control-plane health probe. User-operation callers must validate
    their trusted identity before invoking it; readiness never authorizes data.
    """
    global _readiness_cache
    now = time.monotonic()
    base_url = get_memory_service_url()
    service_key = get_memory_service_key()
    config_key = (base_url, hashlib.sha256((service_key or "").encode()).hexdigest())
    if not refresh:
        with _readiness_cache_lock:
            cached = _readiness_cache
        if (
            cached is not None
            and cached[1] == config_key
            and now - cached[0] < READINESS_CACHE_SECONDS
        ):
            return cached[2]

    result = _probe_readyz(session, base_url, service_key)
    with _readiness_cache_lock:
        _readiness_cache = (now, config_key, result)
    return result


def remote_memory_ready(
    *,
    refresh: bool = False,
    session: Optional[requests.Session] = None,
) -> tuple[bool, Optional[str]]:
    """Compatibility wrapper: usable storage, with a reason even if degraded.

    A true flag does not promise semantic search. Consumers needing capabilities
    must use ``remote_memory_readiness``.
    """
    result = remote_memory_readiness(refresh=refresh, session=session)
    return result.available, result.reason


def _probe_readyz(
    session: Optional[requests.Session],
    base_url: str | None,
    service_key: str | None,
) -> RemoteMemoryReadiness:
    """Classify the server's status/db/embedding contract without exposing details."""
    if not base_url:
        return RemoteMemoryReadiness(
            "unavailable", reason=READINESS_UNCONFIGURED_REASON
        )
    headers = {"Authorization": f"Bearer {service_key}"} if service_key else {}
    headers["User-Agent"] = MEMORY_SERVICE_USER_AGENT
    try:
        response = (session if session is not None else _get_shared_session()).get(
            f"{base_url.rstrip('/')}{READY_PATH}",
            headers=headers,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
    except requests.exceptions.RequestException:
        return RemoteMemoryReadiness("unavailable", reason=READINESS_UNAVAILABLE_REASON)
    if not 200 <= response.status_code < 300:
        return RemoteMemoryReadiness("unavailable", reason=READINESS_UNAVAILABLE_REASON)
    invalid = RemoteMemoryReadiness(
        "protocol_error", reason=READINESS_PROTOCOL_ERROR_REASON
    )
    if response.status_code != 200:
        return invalid
    try:
        payload = response.json()
    except ValueError:
        return invalid
    if not isinstance(payload, dict):
        return invalid
    status, db, embedding = (
        payload.get("status"),
        payload.get("db"),
        payload.get("embedding"),
    )
    if (
        not isinstance(status, str)
        or not isinstance(db, str)
        or status not in {"ok", "degraded", "unavailable"}
        or db not in {"ok", "error"}
    ):
        return invalid
    if not isinstance(embedding, dict):
        return invalid
    embedding_status = embedding.get("status")
    degraded = embedding.get("degraded")
    if (
        not isinstance(embedding_status, str)
        or embedding_status not in {"ok", "degraded"}
        or type(degraded) is not bool
    ):
        return invalid
    if degraded != (embedding_status == "degraded"):
        return invalid
    if db == "error" or status == "unavailable":
        return RemoteMemoryReadiness("unavailable", reason=READINESS_UNAVAILABLE_REASON)
    if (status == "degraded") != degraded:
        return invalid
    if degraded:
        return RemoteMemoryReadiness("degraded", reason=READINESS_DEGRADED_REASON)
    return RemoteMemoryReadiness("ready", semantic_available=True)


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
    if raw_content is None and data.get("content_b64") is not None:
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

    def _validate_identity_binding(self, binding: _ParentIdentityBinding) -> None:
        """An observed validation failure permanently ends this binding.

        Even a missing identity must pass through this boundary. Restoring the
        entry user/config/publication cannot revive an invalidated scope; only
        an explicitly resolved new scope can use the restored context.
        """
        if binding.invalidated.is_set():
            raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON)
        try:
            self._check_identity_binding(binding)
        except RemoteMemoryStoreIdentityError:
            binding.invalidated.set()
            raise

    def _check_identity_binding(self, binding: _ParentIdentityBinding) -> None:
        """Recheck the captured user/mode/config/publication before I/O.

        Re-publication or expiry invalidates this scope even if the new mapping
        still includes the same seat. The caller must resolve a new scope;
        this method never refreshes, retries or switches to the parent identity.
        Already dispatched requests cannot be retroactively cancelled.
        """
        identity = require_remote_memory_identity(self._user_id_provider())
        url, fingerprint = _configured_identity_target()
        if (
            identity != binding.user_external_id
            or get_memory_agent_seats_enabled()
            is not isinstance(binding, _AGENT_BINDING_TYPES)
            or url != binding.service_url
            or self._base_url != binding.service_url
            or fingerprint != binding.credential_fingerprint
            or hashlib.sha256(self._service_key.encode()).hexdigest()
            != binding.credential_fingerprint
        ):
            raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON)
        if not isinstance(binding, _AGENT_BINDING_TYPES):
            return
        if binding.owner_check is not None:
            try:
                binding.owner_check()
            except RemoteMemoryStoreIdentityError:
                raise
            except Exception:  # noqa: BLE001 - unverifiable ownership ends the scope
                raise RemoteMemoryStoreIdentityError(
                    AGENT_SEAT_REQUIRED_REASON
                ) from None
        if isinstance(binding, _ConfiguredAgentIdentityBinding):
            if binding.owner_check is None:
                raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON)
            try:
                current = find_binding(
                    identity, int(binding.agent_ref), url, fingerprint
                )
            except (ValueError, TypeError, RecursionError, OverflowError):
                raise RemoteMemoryStoreIdentityError(
                    AGENT_SEAT_REQUIRED_REASON
                ) from None
            if current != binding.mapping:
                raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON)
            return
        snapshot, epoch = _seat_snapshot(identity)
        if (
            epoch != binding.provider_generation
            or snapshot != binding.snapshot
            or binding.agent_ref not in snapshot.agent_refs
        ):
            raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON)

    def _check_configured_mapping_selection(
        self, binding: _ConfiguredAgentIdentityBinding
    ) -> None:
        """Check existing selectors against trusted initialization IDs once.

        whoami does not attest agent_owner_id. That relation must come from
        the administrator's existing provisioning record in binding.mapping.
        No organization enumeration, scope expansion or identity retry occurs.
        """
        for external, expected in (
            (binding.user_external_id, binding.mapping.memory_user_id),
            (f"agent:{binding.agent_ref}", binding.mapping.memory_agent_id),
        ):
            try:
                response = self._session.request(
                    "GET",
                    f"{self._base_url}{WHOAMI_PATH}",
                    headers={
                        "Authorization": f"Bearer {self._service_key}",
                        "X-User-External-Id": external,
                        "User-Agent": MEMORY_SERVICE_USER_AGENT,
                    },
                    timeout=(self._connect_timeout, self._read_timeout),
                    allow_redirects=False,
                )
                payload = response.json() if response.status_code == 200 else None
                identity = (
                    payload.get("identity") if isinstance(payload, dict) else None
                )
                scopes = payload.get("scopes") if isinstance(payload, dict) else None
                if (
                    not isinstance(identity, dict)
                    or type(identity.get("id")) is not int
                    or identity["id"] != expected
                    or not isinstance(scopes, list)
                    or any(type(scope) is not str for scope in scopes)
                    or set(scopes) != {"memory:read", "memory:write"}
                ):
                    raise RemoteMemoryStoreIdentityError(AGENT_SEAT_REQUIRED_REASON)
            except (requests.RequestException, ValueError):
                raise RemoteMemoryStoreIdentityError(
                    AGENT_SEAT_REQUIRED_REASON
                ) from None

    def validate_identity_context(self) -> str:
        """Locally validate a user operation before data or user-health I/O."""
        binding = _agent_identity_override.get()
        if binding is not None:
            self._validate_identity_binding(binding)
            return (
                f"agent:{binding.agent_ref}"
                if isinstance(binding, _AGENT_BINDING_TYPES)
                else binding.user_external_id
            )
        return require_remote_memory_identity(self._user_id_provider())

    def _headers(self) -> dict[str, str]:
        """Build request headers (§3.4).

        The service key authenticates the process; ``X-User-External-Id``
        carries the acting xagent user so the service can resolve ownership.
        Inside :func:`delegated_agent_identity` the header carries the seat
        identity (``agent:<ref>``) instead. A missing trusted user id raises
        before any HTTP, including when a seat override exists.
        """
        headers = {
            "Authorization": f"Bearer {self._service_key}",
            "Content-Type": "application/json",
            "User-Agent": MEMORY_SERVICE_USER_AGENT,
        }
        headers["X-User-External-Id"] = self.validate_identity_context()
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
            response = self._session.request(
                method,
                url,
                json=body,
                params=params,
                headers=self._headers(),
                timeout=timeout,
            )
        except requests.exceptions.Timeout as error:
            _record_transport_failure(
                path, f"{'写' if write else '读'}超时 {read_timeout:.1f}s"
            )
            raise RemoteMemoryStoreError(
                f"Remote memory upstream unreachable (request timed out after "
                f"{read_timeout:.1f}s -- a transport failure, not a credential "
                f"rejection): {method} {path}"
            ) from error
        except requests.exceptions.RequestException as error:
            _record_transport_failure(path, f"连接失败（{type(error).__name__}）")
            raise RemoteMemoryStoreError(
                f"Remote memory upstream unreachable (connection failure -- not "
                f"a credential rejection): {method} {path} — {error}"
            ) from error
        if response.status_code == 401:
            binding = _agent_identity_override.get()
            if isinstance(binding, _AGENT_BINDING_TYPES):
                # Fence before response instrumentation can inspect an invalid
                # body. A malformed rejection cannot leave the old scope live.
                binding.invalidated.set()
        _record_upstream_answer(path, response)
        return response

    @staticmethod
    def _json(path: str, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise RemoteMemoryStoreError(
                f"Remote memory service returned a non-JSON response: {path}"
            ) from error

    def _raise_for_error(self, path: str, response: requests.Response) -> None:
        """Raise unless the service answered with a 2xx (§3.5.1).

        Independent scopes are permanently fenced on 401, without diagnostic
        identity switching. Other 401s are classified instead of blaming the
        credential by default (缺陷一). The upstream answers a bad key,
        revoked key and unknown ``X-User-External-Id`` with the same generic
        ``{"error":{"code":"unauthorized","message":"unauthorized"}}`` body
        (Go ``internal/app/identity/service.go:28``), so the raised text is
        decided from observable evidence: credential wording in the upstream
        message, or a bare-key ``/v1/whoami`` probe when an identity header
        was sent. Every 401 text carries the HTTP status, the upstream
        ``error.code``/``error.message`` and the actionable next step.
        """
        status = response.status_code
        if 200 <= status < 300:
            return
        binding = _agent_identity_override.get()
        independent_rejected = status == 401 and isinstance(
            binding, _AGENT_BINDING_TYPES
        )
        if independent_rejected:
            # A selected identity/key has been rejected. Fence before parsing
            # the response, and do not retry or diagnose under a bare identity.
            assert binding is not None
            binding.invalidated.set()
        # the service's own message carries the repair ("send a follow-up delivery
        # instead" for a 409) — dropping it left the model retrying blind
        code: Optional[str] = None
        message = ""
        try:
            payload = response.json()
            error = payload.get("error") if isinstance(payload, dict) else None
            error = error if isinstance(error, dict) else {}
            if isinstance(error.get("code"), str):
                code = error["code"]
            if isinstance(error.get("message"), str):
                message = error["message"]
        except (ValueError, RecursionError):
            pass
        if status == 401:
            if independent_rejected:
                code_note = f"（code={code}）" if code else ""
                message_note = f"，上游 message: {message}" if message else ""
                raise RemoteMemoryStoreAuthenticationError(
                    f"上游拒绝了当前独立身份或凭据：HTTP 401{code_note} on "
                    f"{path}{message_note}。当前记忆 scope 已失效；"
                    "请核对上游身份/凭据，并重新进入独立 scope。",
                    status_code=status,
                )
            raise RemoteMemoryStoreAuthenticationError(
                self._classify_401(path, code, message),
                status_code=status,
            )
        code_note = f"（code={code}）" if code else ""
        detail = f"，上游 message: {message}" if message else ""
        raise RemoteMemoryStoreError(
            f"Remote memory service returned HTTP {status}{code_note} on "
            f"{path}{detail}",
            status_code=status,
        )

    def _classify_401(self, path: str, code: Optional[str], message: str) -> str:
        """Explain one 401 as honestly as the evidence allows.

        Three outcomes, in evidence order:

        * the upstream message names the credential itself → report a key
          rejection and skip the probe;
        * an identity header was sent → one cached bare-key
          :data:`WHOAMI_PATH` probe separates "key invalid" (probe 401) from
          "identity not provisioned" (probe 200);
        * the probe could not decide → say so and name both hypotheses
          instead of asserting either one.
        """
        code_note = f"（code={code}）" if code else ""
        message_note = f"，上游 message: {message}" if message else ""
        lowered = f"{code or ''} {message}".lower()
        if any(
            marker in lowered
            for marker in ("credential", "api key", "api_key", "token", "key")
        ):
            return (
                f"上游拒绝了服务凭据：HTTP 401{code_note} on {path}{message_note}。"
                f"上游明示凭据类错误，请核对 XAGENT_MEMORY_SERVICE_KEY"
                f"（无效、已吊销或不属于该实例）。"
            )
        external_id = self._headers().get("X-User-External-Id")
        if external_id is not None:
            valid, note = self._probe_key_validity()
            if valid is True:
                return (
                    f"身份未开通：HTTP 401{code_note} on {path}{message_note}。"
                    f"服务 key 本身有效（{WHOAMI_PATH} 探测 200），但 "
                    f"external_id={external_id} 在上游身份目录不存在或未激活，"
                    f"需在 memsrv org 建号后重试。"
                )
            if valid is False:
                return (
                    f"上游拒绝了服务凭据：HTTP 401{code_note} on {path}{message_note}。"
                    f"key 无效（{WHOAMI_PATH} 探测同样 401），请核对 "
                    f"XAGENT_MEMORY_SERVICE_KEY。"
                )
            return (
                f"上游返回 HTTP 401{code_note} on {path}{message_note}，"
                f"且诊断探测未能完成（{note}）：可能是 key 无效/被拒，"
                f"也可能是 external_id={external_id} 未在 memsrv 开通。"
            )
        return (
            f"上游返回 HTTP 401{code_note} on {path}{message_note}："
            f"本次请求未携带 external_id，上游拒绝的是服务凭据本身"
            f"（key 无效、已吊销或不属于该实例）。"
        )

    def _probe_key_validity(self) -> tuple[Optional[bool], str]:
        """Judge the key with one cached bare ``/v1/whoami`` call.

        Returns ``(True, note)`` when the key is accepted without any
        identity header, ``(False, note)`` when even the bare call is 401, and
        ``(None, note)`` when the probe cannot decide (transport failure or a
        non-401 error status). Only definitive verdicts are cached; the probe
        never raises, and it never touches the upstream health signal -- it is
        a diagnostic, not business traffic.
        """
        fingerprint = hashlib.sha256(self._service_key.encode("utf-8")).hexdigest()[:16]
        now = time.monotonic()
        with _key_validity_cache_lock:
            cached = _key_validity_cache.get(fingerprint)
        if cached is not None and now - cached[0] < KEY_VALIDITY_CACHE_SECONDS:
            return cached[1], "按缓存判定"
        try:
            response = self._session.request(
                "GET",
                f"{self._base_url}{WHOAMI_PATH}",
                headers={
                    "Authorization": f"Bearer {self._service_key}",
                    "User-Agent": MEMORY_SERVICE_USER_AGENT,
                },
                timeout=(self._connect_timeout, self._read_timeout),
            )
        except requests.exceptions.RequestException as error:
            return None, f"探测请求失败（{type(error).__name__}: {error}）"
        verdict: Optional[bool]
        if 200 <= response.status_code < 300:
            verdict = True
        elif response.status_code == 401:
            verdict = False
        else:
            return None, f"探测返回 HTTP {response.status_code}，无法判定"
        with _key_validity_cache_lock:
            _key_validity_cache[fingerprint] = (now, verdict)
        return verdict, "探测已完成"

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
        """Forward caller filters, applying scope rules only when asked.

        Dimensions are stamp-only by default: a scope's ``memory_dimensions``
        are written onto notes (add/remember) but never merged into search
        filters implicitly -- a task-scoped writer would otherwise search only
        its own task's notes, blinding cross-task recall and dedup. Callers
        opt in with ``filters={SCOPE_DIMENSIONS_FILTER_KEY: True}``; the key
        is stripped before the request is sent. ``strict_memory_isolation``
        without dimensions keeps its ``__scope_exclusive__`` directive
        unchanged (cooperative one-way visibility). As before, no ``user_id``
        filter is ever added: ownership is the service's job (§3.2).
        """
        scope = get_execution_scope()
        merged = dict(filters) if filters else {}
        wants_dimensions = bool(merged.pop(SCOPE_DIMENSIONS_FILTER_KEY, False))
        dimension_metadata = (
            memory_dimension_metadata(scope)
            if wants_dimensions and scope is not None
            else {}
        )
        exclusive = bool(
            scope is not None
            and scope.strict_memory_isolation
            and not scope.memory_dimensions
        )
        if not dimension_metadata and not exclusive:
            return merged or None

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
        """Whether the active scope constrains by-id (get/update/delete).

        Only a strict dimension-less scope gates by id (the one-way
        visibility rule). Dimensions are stamp-only cooperative tags, so a
        dimensioned scope never narrows by-id access; ownership stays the
        service's job.
        """
        scope = get_execution_scope()
        if scope is None:
            return False
        return bool(scope.strict_memory_isolation) and not scope.memory_dimensions

    @staticmethod
    def _scope_permits(note: MemoryNote) -> bool:
        """Whether the active scope may touch ``note`` by id.

        A strict dimension-less scope keeps one-way visibility: a note
        stamped with any scope dimension is invisible to it. A dimensioned
        scope does not narrow by-id access (dimensions are stamp-only,
        cooperative); ownership is enforced by the service.
        """
        scope = get_execution_scope()
        if (
            scope is not None
            and scope.strict_memory_isolation
            and not scope.memory_dimensions
        ):
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

        ``tags`` is always sent, even when empty: the service keeps whatever
        list a PUT carries (or refills an empty one from ``keywords``), so
        omitting the field would silently preserve the old tags after a content
        rewrite (X-06, probed 2026-10-29).
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
    def _note_patch_payload(note: MemoryNote) -> dict[str, Any]:
        """Serialize ordinary edits without identity or structural columns.

        An omitted model field remains omitted; explicitly supplied empty lists
        or metadata remain meaningful clears. Nonempty in-place edits to default
        mutable fields are included. Protected metadata is preserved verbatim so
        the server can accept unchanged values and reject attempted changes.
        This is not a client-revision compare-and-swap contract.
        """
        fields = note.model_fields_set
        payload: dict[str, Any] = {}
        if "content" in fields:
            if isinstance(note.content, bytes):
                payload["content_b64"] = base64.b64encode(note.content).decode("ascii")
            else:
                payload["content"] = note.content
        for key in ("keywords", "tags"):
            value = getattr(note, key)
            if key in fields or value:
                payload[key] = list(value)
        for key, default in (("category", "general"), ("mime_type", "text/plain")):
            value = getattr(note, key)
            if key in fields or value != default:
                payload[key] = value
        if "timestamp" in fields:
            payload["timestamp"] = _format_timestamp(note.timestamp)
        if "metadata" in fields or note.metadata:
            payload["metadata"] = dict(note.metadata)
        if not payload:
            raise RemoteMemoryStoreError("Memory update has no editable fields")
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
        response = self._request(
            "PATCH", path, body=self._note_patch_payload(note), write=True
        )
        if response.status_code in (403, 404):
            return self._not_found_or_denied(note.id)
        self._raise_for_error(path, response)
        acknowledgement = self._json(path, response)
        if (
            not isinstance(acknowledgement, dict)
            or acknowledgement.get("success") is not True
            or acknowledgement.get("id") != note.id
        ):
            raise RemoteMemoryStoreError(
                "Remote memory PATCH returned an invalid acknowledgement"
            )
        return MemoryResponse(
            success=True,
            memory_id=note.id,
            metadata={
                key: acknowledgement[key]
                for key in (
                    "owner_id",
                    "space_kind",
                    "space_id",
                    "modified_by",
                    "degraded",
                )
                if key in acknowledgement
            },
        )

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
        so the caller can report what happened instead of guessing. The
        service's own ``degraded`` flag rides along when the payload carries
        it, so a partially applied write is not reported as a clean one.
        Failures keep the adapter's usual semantics (§3.5.1): transport errors
        and rejections raise, so a write that did not happen is never reported
        as stored.
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
        decision: dict[str, Any] = {
            "action": action,
            "parent_id": payload.get("parent_id"),
        }
        if "degraded" in payload:
            decision["degraded"] = payload.get("degraded")
        return MemoryResponse(
            success=True,
            memory_id=str(payload.get("id") or note.id),
            metadata=decision,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[MemoryNote]:
        path = "/v1/memory/search"
        # An absent threshold used to travel as ``null``, which the service read
        # as "no ceiling" and answered with most of the store. Fall back to the
        # measured distance ceiling instead so "no opinion" still means "only
        # what is actually close".
        threshold = (
            DEFAULT_SEARCH_SIMILARITY_THRESHOLD
            if similarity_threshold is None
            else similarity_threshold
        )
        body = {
            "query": query,
            "k": k,
            "filters": self._filters_payload(filters),
            "similarity_threshold": threshold,
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
    "WHOAMI_PATH",
    "RemoteMemoryStore",
    "RemoteMemoryStoreAuthenticationError",
    "RemoteMemoryStoreConfigurationError",
    "RemoteMemoryStoreError",
    "RemoteMemoryStoreIncompleteError",
    "RemoteMemoryStoreIdentityError",
    "require_remote_memory_identity",
    "WRITE_TIMEOUT_SECONDS",
    "explicit_memory_owner",
    "remote_memory_ready",
    "remote_memory_readiness",
    "RemoteMemoryReadiness",
    "remote_memory_store_from_config",
    "remote_upstream_health",
    "remote_upstream_report",
]
