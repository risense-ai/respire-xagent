"""System-side memsrv verifier. No model/user-facing credential or grant flow.

Only the isolated publisher imports/constructs this connector. Its administrator
pins xagent users to known test memsrv persons once; active owned agents are then
read from xagent's database and revalidated against memsrv automatically. It
never creates users/seats, changes ACLs, reads notes, or follows redirects.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from sqlalchemy import Engine, bindparam, text

from .agent_seats import AgentSeatRequest
from .seat_file_provider import _unique_object
from .seat_publisher import GrantVerificationError, VerifiedSeatGrant


class VerificationUnavailable(GrantVerificationError):
    """Safe operator reason; never includes upstream bodies or secrets."""


@dataclass(frozen=True)
class UserBinding:
    xagent_user_id: int
    memory_user_id: int


def read_publisher_secret(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise VerificationUnavailable(
                "publisher secret must be a private regular file"
            )
        raw = source.read(16_385)
    if not raw or len(raw) > 16_384:
        raise VerificationUnavailable("invalid publisher secret length")
    value = raw.decode().strip()
    if not value or "\n" in value or "\r" in value:
        raise VerificationUnavailable("invalid publisher secret format")
    return value


class MemsrvSeatVerifier:
    """Recheck ownership, policy and actual restricted-key identity selection.

    GET /v1/org/users currently needs org:admin and returns the organization
    snapshot; memsrv has no ID-filtered ownership endpoint. Only pinned records
    enter grants, fingerprints or local results. This limitation must be
    approved for the isolated publisher; runtime keys never receive that scope.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        service_url: str,
        control_key_file: Path,
        runtime_key_file: Path,
        bindings: tuple[UserBinding, ...],
        session: requests.Session | None = None,
    ) -> None:
        parsed = urlsplit(service_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.query
            or parsed.fragment
            or service_url != service_url.strip().rstrip("/")
        ):
            raise ValueError("invalid memory service origin")
        if (
            not bindings
            or any(
                type(b.xagent_user_id) is not int
                or b.xagent_user_id <= 0
                or type(b.memory_user_id) is not int
                or b.memory_user_id <= 0
                for b in bindings
            )
            or len({b.xagent_user_id for b in bindings}) != len(bindings)
            or len({b.memory_user_id for b in bindings}) != len(bindings)
        ):
            raise ValueError("invalid or ambiguous publisher user bindings")
        self.engine, self.service_url = engine, service_url
        self.control_key_file, self.runtime_key_file = (
            control_key_file,
            runtime_key_file,
        )
        self.bindings = bindings
        self.session = session or requests.Session()
        self.session.trust_env = False

    def _get(self, path: str, key: str, identity: str | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {key}"}
        if identity is not None:
            headers["X-User-External-Id"] = identity
        try:
            response = self.session.get(
                self.service_url + path,
                headers=headers,
                timeout=(2, 3),
                allow_redirects=False,
                stream=True,
            )
            try:
                if response.status_code != 200:
                    raise VerificationUnavailable(
                        f"{path} rejected HTTP {response.status_code}"
                    )
                raw = bytearray()
                for block in response.iter_content(16_384):
                    raw.extend(block)
                    if len(raw) > 2_097_152:
                        raise VerificationUnavailable(
                            "organization response exceeds publisher budget"
                        )
                result = json.loads(raw, object_pairs_hook=_unique_object)
                if type(result) is not dict:
                    raise VerificationUnavailable("invalid memsrv response object")
                return result
            finally:
                response.close()
        except (requests.RequestException, OSError, ValueError) as error:
            if isinstance(error, VerificationUnavailable):
                raise
            raise VerificationUnavailable("memsrv verification unavailable") from None

    def _database_snapshot(self) -> dict[str, Any]:
        ids = [b.xagent_user_id for b in self.bindings]
        with self.engine.connect() as connection, connection.begin():
            if self.engine.dialect.name == "postgresql":
                connection.exec_driver_sql(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
                )
            elif self.engine.dialect.name == "sqlite":
                connection.exec_driver_sql("PRAGMA query_only=ON")
            else:
                raise VerificationUnavailable("unsupported publisher database dialect")
            users = (
                connection.execute(
                    text(
                        "SELECT id,is_admin FROM users WHERE id IN :ids ORDER BY id"
                    ).bindparams(bindparam("ids", expanding=True)),
                    {"ids": ids},
                )
                .mappings()
                .all()
            )
            agents = (
                connection.execute(
                    text(
                        "SELECT id,user_id,status,team_id,visibility FROM agents WHERE user_id IN :ids AND status IN ('draft','published') ORDER BY id"
                    ).bindparams(bindparam("ids", expanding=True)),
                    {"ids": ids},
                )
                .mappings()
                .all()
            )
        if {row["id"] for row in users} != set(ids):
            raise VerificationUnavailable("pinned xagent user missing")
        if any(row["team_id"] is not None for row in agents):
            raise VerificationUnavailable(
                "team-owned agents need an explicit host authorization resolver"
            )
        return {
            "users": [dict(row) for row in users],
            "agents": [dict(row) for row in agents],
        }

    @staticmethod
    def _org_rows(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
        rows = document.get("users")
        if type(rows) is not list:
            raise VerificationUnavailable("invalid organization snapshot")
        result = {}
        for row in rows:
            if (
                type(row) is not dict
                or type(row.get("id")) is not int
                or row["id"] <= 0
                or row["id"] in result
            ):
                raise VerificationUnavailable("invalid organization identity")
            result[row["id"]] = row
        return result

    def _select_identity(self, key: str, external: str, expected_id: int) -> None:
        result = self._get("/v1/whoami", key, external)
        identity = result.get("identity")
        if (
            type(identity) is not dict
            or type(identity.get("id")) is not int
            or identity.get("id") != expected_id
            or identity.get("role") != "member"
            or result.get("is_admin") is not False
            or result.get("kind") not in {"user", "service"}
            or type(result.get("scopes")) is not list
            or any(type(scope) is not str for scope in result["scopes"])
            or set(result["scopes"]) != {"memory:read", "memory:write"}
        ):
            raise VerificationUnavailable(
                "runtime identity or permission profile mismatch"
            )

    def __call__(self) -> tuple[VerifiedSeatGrant, ...]:
        control = read_publisher_secret(self.control_key_file)
        runtime = read_publisher_secret(self.runtime_key_file)
        if control == runtime:
            raise VerificationUnavailable("control and runtime credentials must differ")
        before_db = self._database_snapshot()
        before_org = self._org_rows(self._get("/v1/org/users", control))
        grants = []
        selected_ids = set()
        profile = hashlib.sha256(runtime.encode()).hexdigest()
        for binding in self.bindings:
            person = before_org.get(binding.memory_user_id)
            if (
                person is None
                or person.get("external_id") != str(binding.xagent_user_id)
                or person.get("kind") != "person"
                or person.get("status") != "active"
                or person.get("role") != "member"
            ):
                raise VerificationUnavailable(
                    "pinned memory person mapping unavailable"
                )
            self._select_identity(
                runtime, str(binding.xagent_user_id), binding.memory_user_id
            )
            selected_ids.add(binding.memory_user_id)
            owned = [
                row
                for row in before_db["agents"]
                if row["user_id"] == binding.xagent_user_id
            ]
            seats = []
            for agent in owned:
                matches = [
                    row
                    for row in before_org.values()
                    if row.get("kind") == "agent"
                    and row.get("agent_ref") == str(agent["id"])
                ]
                if len(matches) != 1:
                    raise VerificationUnavailable(
                        "owned xagent agent needs a unique provisioned seat"
                    )
                seat = matches[0]
                if (
                    seat.get("status") != "active"
                    or seat.get("role") != "member"
                    or seat.get("agent_owner_id") != binding.memory_user_id
                ):
                    raise VerificationUnavailable(
                        "memory seat ownership or active status mismatch"
                    )
                self._select_identity(runtime, f"agent:{agent['id']}", seat["id"])
                selected_ids.add(seat["id"])
                seats.append(seat)
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "person": person,
                        "seats": seats,
                        "xagent": {
                            "users": [
                                row
                                for row in before_db["users"]
                                if row["id"] == binding.xagent_user_id
                            ],
                            "agents": owned,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            grants.append(
                VerifiedSeatGrant(
                    AgentSeatRequest(
                        self.service_url, str(binding.xagent_user_id), profile
                    ),
                    frozenset(str(agent["id"]) for agent in owned),
                    fingerprint,
                )
            )
        # No common revision exists across systems. Reject observed drift across
        # both reads; do not claim an atomic cross-database authorization snapshot.
        after_org = self._org_rows(self._get("/v1/org/users", control))
        if (
            any(after_org.get(id_) != before_org.get(id_) for id_ in selected_ids)
            or self._database_snapshot() != before_db
        ):
            raise VerificationUnavailable("authorization changed during verification")
        return tuple(grants)
