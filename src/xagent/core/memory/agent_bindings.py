"""Small trusted initialization mapping, not a runtime permission publisher.

An administrator must obtain the person/seat IDs and ownership from existing
provisioning records. Equality in this config is an attestation by that trusted
administrator, not proof derived from whoami or from permission to execute an
agent. Nothing here provisions users, enumerates an organization or grants new
memsrv rights. Missing/unusable records fail closed; no lease or background I/O.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ...config import get_memory_agent_bindings


@dataclass(frozen=True)
class ApprovedAgentBinding:
    user_id: int
    agent_id: int
    memory_user_id: int
    memory_agent_id: int
    memory_agent_owner_id: int
    service_url: str
    runtime_profile: str


def approved_bindings() -> tuple[ApprovedAgentBinding, ...] | None:
    raw = get_memory_agent_bindings()
    if raw is None:
        return None  # preserve explicitly installed legacy trusted host providers
    if len(raw.encode()) > 262_144:
        raise ValueError("agent binding config too large")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate agent binding field")
            result[key] = value
        return result

    rows = json.loads(raw, object_pairs_hook=unique)
    if type(rows) is not list or len(rows) > 1000:
        raise ValueError("invalid agent binding config")
    result = []
    identities = set()
    seats = set()
    refs = set()
    for row in rows:
        if type(row) is not dict or set(row) != set(
            ApprovedAgentBinding.__dataclass_fields__
        ):
            raise ValueError("invalid agent binding fields")
        if any(
            type(row[key]) is not int or row[key] <= 0
            for key in (
                "user_id",
                "agent_id",
                "memory_user_id",
                "memory_agent_id",
                "memory_agent_owner_id",
            )
        ):
            raise ValueError("invalid agent binding identity")
        if row["memory_agent_owner_id"] != row["memory_user_id"]:
            raise ValueError("agent binding ownership mismatch")
        url, profile = row["service_url"], row["runtime_profile"]
        if (
            type(url) is not str
            or not url
            or url != url.strip().rstrip("/")
            or type(profile) is not str
            or len(profile) != 64
            or any(c not in "0123456789abcdef" for c in profile)
        ):
            raise ValueError("invalid agent binding target")
        identity = row["user_id"], row["agent_id"], url, profile
        if identity in identities:
            raise ValueError("ambiguous agent binding")
        seat = url, profile, row["memory_agent_id"]
        ref = url, profile, row["agent_id"]
        if seat in seats or ref in refs:
            raise ValueError("ambiguous agent seat binding")
        identities.add(identity)
        seats.add(seat)
        refs.add(ref)
        result.append(ApprovedAgentBinding(**row))
    return tuple(result)


def find_binding(
    user_id: str, agent_id: int, service_url: str, profile: str
) -> ApprovedAgentBinding | None:
    bindings = approved_bindings()
    if bindings is None:
        return None
    for binding in bindings:
        if (
            str(binding.user_id),
            binding.agent_id,
            binding.service_url,
            binding.runtime_profile,
        ) == (user_id, agent_id, service_url, profile):
            return binding
    raise ValueError("approved agent binding missing")
