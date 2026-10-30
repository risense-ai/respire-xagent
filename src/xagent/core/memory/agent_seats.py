"""Trusted, local publication of provisioned agent-seat mappings.

The host owns provisioning and refresh. It may fetch a registry with separate
control-plane credentials, then publish an immutable snapshot here. Runtime
consumers never receive those credentials or call an organization endpoint.
Providers must read local state without I/O. Every process must install its own
provider; missing, expired or revoked publications fail closed. A mapping selects
an identity only; the memory service still authorizes every operation.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSeatRequest:
    service_url: str
    user_external_id: str
    runtime_profile: str


@dataclass(frozen=True)
class AgentSeatSnapshot:
    """Verified active grant set; valid_until uses the local monotonic clock."""

    request: AgentSeatRequest
    agent_refs: frozenset[str]
    generation: int
    valid_until: float
    status: str
    renewable: bool = field(default=False, kw_only=True)

    def continues(self, previous: AgentSeatSnapshot) -> bool:
        """Only an explicitly continuous lease may change its deadline.

        Renewable providers must fence a generation on revocation, authority
        changes, expiry gaps and local failures, including unobserved gaps.
        Legacy/custom publications retain exact-snapshot comparison.
        """
        return self == previous or (
            self.renewable
            and previous.renewable
            and self.request == previous.request
            and self.agent_refs == previous.agent_refs
            and self.generation == previous.generation
            and self.status == previous.status
        )


AgentSeatProvider = Callable[[AgentSeatRequest], AgentSeatSnapshot | None]
_provider: AgentSeatProvider | None = None
_provider_generation = 0
_provider_lock = threading.Lock()


def set_trusted_agent_seat_provider(provider: AgentSeatProvider | None) -> None:
    """Host-only installation; replacing/clearing a provider revokes old scopes.

    Never wire this setter to request headers, task payloads or untrusted tools.
    A deployment must refresh publications before expiry in every worker. The
    adapter supplies no automatic provisioning, credential acquisition or I/O.
    """
    if provider is not None and not callable(provider):
        raise TypeError("agent seat provider must be callable")
    global _provider, _provider_generation
    with _provider_lock:
        _provider = provider
        _provider_generation += 1


def read_agent_seat_snapshot(
    request: AgentSeatRequest,
) -> tuple[AgentSeatSnapshot, int]:
    """Read one current local publication, rejecting invalid provider results."""
    with _provider_lock:
        provider, epoch = _provider, _provider_generation
    if provider is None:
        raise ValueError("agent seat provider missing")
    snapshot = provider(request)
    if (
        not isinstance(snapshot, AgentSeatSnapshot)
        or snapshot.request != request
        or snapshot.status != "active"
        or type(snapshot.renewable) is not bool
        or type(snapshot.agent_refs) is not frozenset
        or any(
            not isinstance(ref, str)
            or not ref.isascii()
            or not ref.isdecimal()
            or int(ref) <= 0
            or str(int(ref)) != ref
            for ref in snapshot.agent_refs
        )
        or type(snapshot.generation) is not int
        or snapshot.generation <= 0
        or type(snapshot.valid_until) not in (int, float)
        or not math.isfinite(snapshot.valid_until)
        or time.monotonic() >= snapshot.valid_until
    ):
        raise ValueError("agent seat publication invalid or expired")
    with _provider_lock:
        if epoch != _provider_generation or provider is not _provider:
            raise ValueError("agent seat provider changed")
    return snapshot, epoch


class TrustedAgentSeatRegistry:
    """Optional in-process provider for a host's verified control-plane results.

    Publishers explicitly scope mappings by service URL and trusted User.id;
    the runtime authority's SHA-256 fingerprint explicitly identifies the target
    profile. Control-plane credentials are absent and must not be used to derive
    that runtime profile. Publishers must verify each seat is active and that
    this user/profile may address it; org-wide membership alone is insufficient.
    Each publish gets a new local generation, even if the seats are unchanged, so existing scopes must end
    and resolve a fresh mapping. Clearing a mapping immediately denies new
    requests in active scopes when their next preflight observes the change.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._snapshots: dict[AgentSeatRequest, AgentSeatSnapshot] = {}

    def publish(
        self,
        *,
        service_url: str,
        user_id: int,
        runtime_profile: str,
        agent_ids: Iterable[int],
        ttl_seconds: float,
        status: str,
    ) -> AgentSeatSnapshot:
        if (
            not isinstance(service_url, str)
            or not service_url.strip()
            or type(user_id) is not int
            or user_id <= 0
        ):
            raise ValueError("invalid agent seat publication target")
        if (
            not isinstance(runtime_profile, str)
            or len(runtime_profile) != 64
            or any(char not in "0123456789abcdef" for char in runtime_profile)
        ):
            raise ValueError("invalid runtime profile fingerprint")
        if status != "active":
            raise ValueError("agent seat publication must attest active status")
        refs = set()
        for agent_id in agent_ids:
            if type(agent_id) is not int or agent_id <= 0:
                raise ValueError("invalid agent seat reference")
            refs.add(str(agent_id))
        if (
            type(ttl_seconds) not in (int, float)
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("invalid agent seat publication lifetime")
        request = AgentSeatRequest(
            service_url.strip().rstrip("/"), str(user_id), runtime_profile
        )
        with self._lock:
            self._generation += 1
            snapshot = AgentSeatSnapshot(
                request,
                frozenset(refs),
                self._generation,
                time.monotonic() + ttl_seconds,
                status,
            )
            self._snapshots[request] = snapshot
            return snapshot

    def revoke(self, *, service_url: str, user_id: int, runtime_profile: str) -> None:
        with self._lock:
            self._snapshots.pop(
                AgentSeatRequest(
                    service_url.rstrip("/"), str(user_id), runtime_profile
                ),
                None,
            )

    def __call__(self, request: AgentSeatRequest) -> AgentSeatSnapshot | None:
        with self._lock:
            return self._snapshots.get(request)
