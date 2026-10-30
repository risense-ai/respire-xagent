#!/usr/bin/env python3
"""Live memory round-trip against a running memsrv.

Drives the xagent remote-memory adapter against a real service and reports
which memory surfaces the service actually supports. Exit code 0 means every
required path passed; known gaps (missing PATCH, refused user deletion) are
reported as failures with a hint, never silently tolerated.

Environment:
  XAGENT_MEMORY_SERVICE_URL   base URL, e.g. http://127.0.0.1:8200
  XAGENT_MEMORY_SERVICE_KEY   service key (Bearer)
  XAGENT_MEMORY_IDENTITY      platform user id sent as X-User-External-Id

Options:
  --create-probe-user   create a temporary user with that external_id first
                        (needs an admin key), delete it afterwards
  --keep-note           leave the probe note in place instead of deleting it
  --json                print a machine-readable summary only

The script writes exactly one note (id ``live-probe-<unix-ts>``) into the
identity's memory and deletes it unless ``--keep-note`` is given. It never
touches other users' data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if (REPO_ROOT / "src").is_dir():
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xagent.core.memory.core import MemoryNote  # noqa: E402
from xagent.core.memory.remote import RemoteMemoryStore  # noqa: E402
from xagent.core.user_context import current_user_id  # noqa: E402


def _result(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"check": name, "ok": ok, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-probe-user", action="store_true")
    parser.add_argument("--keep-note", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    url = os.environ.get("XAGENT_MEMORY_SERVICE_URL", "").rstrip("/")
    key = os.environ.get("XAGENT_MEMORY_SERVICE_KEY", "")
    identity = os.environ.get("XAGENT_MEMORY_IDENTITY", "")
    missing = [
        name
        for name, value in (
            ("XAGENT_MEMORY_SERVICE_URL", url),
            ("XAGENT_MEMORY_SERVICE_KEY", key),
            ("XAGENT_MEMORY_IDENTITY", identity),
        )
        if not value
    ]
    if missing:
        sys.exit(f"missing environment: {', '.join(missing)}")
    if not identity.isdigit():
        sys.exit("XAGENT_MEMORY_IDENTITY must be the integer platform user id")

    session = requests.Session()
    headers = {"Authorization": f"Bearer {key}"}
    results: list[dict[str, Any]] = []

    # -- service health ---------------------------------------------------
    for path, name in (("/healthz", "service health"), ("/readyz", "readiness")):
        try:
            response = session.get(f"{url}{path}", timeout=5)
            detail = ""
            if path == "/readyz" and response.ok:
                payload = response.json()
                embedding = payload.get("embedding") or {}
                detail = (
                    f"dim={embedding.get('dim')} model={embedding.get('model')} "
                    f"degraded={embedding.get('degraded')}"
                )
            results.append(
                _result(name, response.ok, detail or f"HTTP {response.status_code}")
            )
        except requests.RequestException as error:
            results.append(_result(name, False, type(error).__name__))

    # -- identity ---------------------------------------------------------
    probe_user_id: int | None = None
    if args.create_probe_user:
        created = session.post(
            f"{url}/v1/org/users",
            headers=headers,
            json={
                "username": f"live-probe-{int(time.time())}",
                "external_id": identity,
                "role": "member",
                "status": "active",
            },
            timeout=10,
        )
        ok = created.ok
        detail = f"HTTP {created.status_code}"
        if ok:
            probe_user_id = created.json().get("id")
            detail = f"user id={probe_user_id}"
        else:
            # A repeated run against the same identity finds the probe user
            # already present; reuse it and do not delete someone else's row.
            existing = session.get(f"{url}/v1/org/users", headers=headers, timeout=10)
            rows = existing.json().get("users") or [] if existing.ok else []
            match = next(
                (
                    row
                    for row in rows
                    if isinstance(row, dict) and str(row.get("external_id")) == identity
                ),
                None,
            )
            if match is not None:
                ok = True
                probe_user_id = None
                detail = f"reused existing user id={match.get('id')}"
        results.append(_result("create probe user", ok, detail))
        if not ok:
            _print(results, args.json)
            return 1
    else:
        users = session.get(f"{url}/v1/org/users", headers=headers, timeout=10)
        found = False
        if users.ok:
            rows = users.json().get("users") or users.json().get("results") or []
            found = any(
                isinstance(row, dict) and str(row.get("external_id")) == identity
                for row in rows
            )
        results.append(
            _result(
                "identity provisioned",
                found,
                f"users.external_id={identity}"
                if found
                else "no user matches; pass --create-probe-user or onboard it (R6)",
            )
        )
        if not found:
            _print(results, args.json)
            return 1

    # -- adapter round-trip ----------------------------------------------
    note_id = f"live-probe-{int(time.time())}"
    token = current_user_id.set(int(identity))
    store = RemoteMemoryStore(url, key, read_timeout=10, write_timeout=30)
    deleted = False
    try:
        try:
            added = store.add(
                MemoryNote(
                    id=note_id,
                    content="live probe: memory round-trip (safe to delete)",
                    category="general",
                    tags=["live-probe"],
                )
            )
            results.append(_result("add", added.success, added.memory_id or ""))
        except Exception as error:  # noqa: BLE001 - report, do not abort
            results.append(_result("add", False, f"{type(error).__name__}: {error}"))

        try:
            got = store.get(note_id)
            content = got.content.content if got.content else None
            results.append(
                _result("get", got.success and bool(content), str(content)[:40])
            )
        except Exception as error:  # noqa: BLE001
            results.append(_result("get", False, f"{type(error).__name__}: {error}"))

        try:
            hits = store.search("memory round-trip probe", k=5)
            ids = [note.id for note in hits]
            results.append(
                _result("search", note_id in ids, f"hits={len(hits)} ids={ids[:3]}")
            )
        except Exception as error:  # noqa: BLE001
            results.append(_result("search", False, f"{type(error).__name__}: {error}"))

        try:
            listed = store.list_all()
            ids = [note.id for note in listed]
            results.append(_result("list_all", note_id in ids, f"notes={len(listed)}"))
        except Exception as error:  # noqa: BLE001
            results.append(
                _result("list_all", False, f"{type(error).__name__}: {error}")
            )

        try:
            updated = store.update(
                MemoryNote(
                    id=note_id, content="live probe: updated body (safe to delete)"
                )
            )
            results.append(_result("update (R1 nonstructural PATCH)", updated.success))
        except Exception as error:  # noqa: BLE001
            detail = f"{type(error).__name__}: {error}"
            if "405" in str(error):
                detail += " -- service has no PATCH route; add PATCH /v1/memory/notes/{id} (R1)"
            results.append(_result("update (R1 nonstructural PATCH)", False, detail))

        if not args.keep_note:
            try:
                deleted = store.delete(note_id).success
                results.append(_result("delete", deleted))
            except Exception as error:  # noqa: BLE001
                results.append(
                    _result("delete", False, f"{type(error).__name__}: {error}")
                )
    finally:
        current_user_id.reset(token)

    # -- cleanup ----------------------------------------------------------
    if probe_user_id is not None:
        removed = session.delete(
            f"{url}/v1/org/users/{probe_user_id}", headers=headers, timeout=10
        )
        detail = f"HTTP {removed.status_code}"
        ok = removed.ok
        if removed.status_code == 500:
            # Known defect: the audit trail blocks deletion. Disable the row so
            # a failed cleanup cannot leave an active probe user behind.
            disabled = session.patch(
                f"{url}/v1/org/users/{probe_user_id}",
                headers=headers,
                json={"status": "disabled"},
                timeout=10,
            )
            detail += (
                " -- audit trail blocks deletion (return 409 or anonymise); "
                f"row disabled instead (HTTP {disabled.status_code})"
            )
        results.append(_result("delete probe user", ok, detail))

    failed = [item for item in results if not item["ok"]]
    _print(results, args.json)
    return 0 if not failed else 1


def _print(results: list[dict[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
        return
    width = max(len(item["check"]) for item in results)
    for item in results:
        status = "PASS" if item["ok"] else "FAIL"
        line = f"{status}  {item['check']:<{width}}  {item['detail']}".rstrip()
        print(line)
    failed = sum(1 for item in results if not item["ok"])
    print()
    print(f"{len(results) - failed} passed, {failed} failed")


if __name__ == "__main__":
    raise SystemExit(main())
