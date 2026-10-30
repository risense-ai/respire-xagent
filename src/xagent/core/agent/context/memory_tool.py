"""Model-invocable memory tools.

ReAct runs with an active memory store expose these tools so the model
manages memories itself during execution — storing valuable insights,
searching for more context, and correcting or removing stale entries —
instead of framework-driven memory pipeline steps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

from ...memory.core import MemoryNote, drop_replaced_tags
from ...memory.remote import (
    NOT_FOUND_OR_DENIED_ERROR,
    RemoteMemoryStoreError,
    _unwrap_remote_store,
)
from ..trace import (
    trace_memory_retrieve_end,
    trace_memory_retrieve_start,
    trace_memory_store_end,
    trace_memory_store_start,
)
from .memory_prompts import (
    CHECK_INBOX_DESCRIPTION,
    DELETE_MEMORY_DESCRIPTION,
    DELIVER_MEMORY_DESCRIPTION,
    SEARCH_MEMORY_DESCRIPTION,
    STORE_MEMORY_DESCRIPTION,
    UPDATE_MEMORY_DESCRIPTION,
)

logger = logging.getLogger(__name__)

STORE_MEMORY_TOOL_NAME = "store_memory"
SEARCH_MEMORY_TOOL_NAME = "search_memory"
UPDATE_MEMORY_TOOL_NAME = "update_memory"
DELETE_MEMORY_TOOL_NAME = "delete_memory"
DELIVER_MEMORY_TOOL_NAME = "deliver_memory"
CHECK_INBOX_TOOL_NAME = "check_inbox"
MEMORY_TOOLS_METADATA_KEY = "memory_tools_enabled"
# Cap per ReAct run. Each DAG step runs its own ReActPattern, so a DAG
# task gets a fresh cap per step rather than one shared task-wide cap.
# 2026-10-14: raised from 5 — a dense planning turn easily states six or
# more durable facts and the silent tail loss read as "auto memory forgot".
DEFAULT_MAX_STORES_PER_RUN = 20
# Cosine distance passed to search(similarity_threshold=...): lower = stricter
# dedup (fewer results count as duplicates).
DEFAULT_DEDUP_DISTANCE_THRESHOLD = 0.1
# Attachment wording refs resolve at most this cosine distance: beyond it the nearest
# note is filler, and a confident wrong quote is worse than an unresolved ref.
ATTACH_WORDING_MAX_DISTANCE = 0.60
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 20
# X-04: a write whose outcome cannot be read back may still have landed. Saying
# so is the only honest answer, and it stops the model from storing the same
# fact again.
#
# Third review: an unconfirmed read-back has two incompatible causes and one
# wording cannot serve both. This message is the *transport* branch only --
# timeout, connection failure or 5xx -- because only there is the enumeration
# true. A read-back the service actually answered (a 401, any other 4xx, or a
# 2xx body that cannot be parsed) was refused or is unreadable, not slow, and
# gets READBACK_REJECTED_MESSAGE below: the shared wording reported a rejected
# credential as a timeout and the model repeated that cause verbatim.
UNVERIFIED_RECEIPT_MESSAGE = (
    "写入已受理但未能确认落库，请勿重复保存 (the write was accepted but could "
    "not be verified -- a read-back timed out, failed to connect or answered "
    "5xx; the memory service may have stored it, so do not save it again)."
)
#: A read-back the service refused (401, or any 4xx outside the 403/404 absence
#: pair) or answered with a body that cannot be read as a note. The write's fate
#: is unknown, and the cause is neither a timeout nor a 5xx.
READBACK_REJECTED_MESSAGE = (
    "读取回执被服务端拒绝或无法解析，写入是否落库未知，请勿重复保存 (the "
    "read-back was refused by the memory service or its answer could not be "
    "parsed; whether the write landed is unknown, so do not save the same fact "
    "again)."
)
#: An unconfirmed read-back whose failure cannot be placed in either class. The
#: honest answer states no cause rather than inventing one.
RECEIPT_FAILURE_UNKNOWN_MESSAGE = (
    "读取回执失败，无法确认写入是否落库，请勿重复保存 (the read-back failed and "
    "its cause could not be determined; whether the write landed is unknown, so "
    "do not save the same fact again)."
)
# X-04 rework: a receipt that arrives without an id is not a not-found answer.
# The write may have landed and simply not been reported back, so it gets its
# own wording instead of being folded into the timeout/5xx message.
MISSING_RECEIPT_ID_MESSAGE = (
    "回执未带 memory id，无法确认是否落库，请勿重复保存 (the service's receipt "
    "carried no memory id, so the write cannot be read back; the memory service "
    "may have stored it, so do not save it again)."
)
# X-04 rework: the recovery created a replacement note, but that note's body
# could not be confirmed to carry the new text. This is not the timeout/5xx
# case above (the create did answer) and must not claim one.
RECOVERED_CONTENT_UNCONFIRMED_MESSAGE = (
    "已新建记忆但未能确认正文一致，请勿重复保存 (a replacement memory was "
    "created because the receipt note did not carry the new text, but the new "
    "note's body could not be confirmed; do not save the same fact again)."
)
# X-04 rework: a create receipt names a live note whose body does not carry the
# text. The write answered, so this is not a timeout either; the note is left
# alone rather than duplicated.
CONTENT_UNCONFIRMED_MESSAGE = (
    "已写入但未能确认正文一致，请勿重复保存 (the note behind the create receipt "
    "exists but its body could not be confirmed to contain the new text; do not "
    "save the same fact again)."
)

MemoryKind = Literal[
    "user_preference",
    "failure_pattern",
    "success_pattern",
    "tool_usage",
    "strategy",
    "domain_insight",
]
_MEMORY_KINDS = frozenset(get_args(MemoryKind))


class StoreMemoryArgs(BaseModel):
    content: str = Field(
        description=(
            "Self-contained memory text. State the insight directly; do not "
            "reference 'this task' or 'the user said above'."
        )
    )
    kind: MemoryKind = Field(
        default="domain_insight",
        description="What kind of insight this memory captures.",
    )


class StoreMemoryTool:
    """Execution-scoped ``store_memory`` tool bound to a memory store."""

    name = STORE_MEMORY_TOOL_NAME
    description = STORE_MEMORY_DESCRIPTION
    args_schema = StoreMemoryArgs

    def __init__(
        self,
        *,
        memory_store: Any,
        task: str,
        runtime: Any | None = None,
        category: str = "react_memory",
        max_stores: int = DEFAULT_MAX_STORES_PER_RUN,
        dedup_distance_threshold: float = DEFAULT_DEDUP_DISTANCE_THRESHOLD,
    ) -> None:
        self.memory_store = memory_store
        self.task = task
        self.runtime = runtime
        self.category = category
        self.max_stores = max_stores
        self.dedup_distance_threshold = dedup_distance_threshold
        self.stored_count = 0
        self._store_lock = asyncio.Lock()

    async def execute(
        self, content: str, kind: str = "domain_insight"
    ) -> dict[str, Any]:
        content = str(content or "").strip()
        if not content:
            return {
                "success": False,
                "error": "Memory content must be a non-empty string.",
            }
        if kind not in _MEMORY_KINDS:
            kind = "domain_insight"

        async with self._store_lock:
            if self.stored_count >= self.max_stores:
                return {
                    "success": False,
                    "error": (
                        f"Memory storage limit ({self.max_stores}) reached for this "
                        "task; do not store further memories."
                    ),
                }

            task_id = str(_runtime_attr(self.runtime, "execution_id") or "")
            step_id = _runtime_attr(self.runtime, "active_react_step_id")
            tracer = _runtime_attr(self.runtime, "tracer")

            if tracer is not None and task_id:
                await trace_memory_store_start(
                    tracer,
                    task_id,
                    data={
                        "task": self.task[:200],
                        "memory_category": self.category,
                        "memory_kind": kind,
                        "source": STORE_MEMORY_TOOL_NAME,
                        "step_id": step_id,
                    },
                )

            duplicate_aware = getattr(self.memory_store, "remember", None)
            if callable(duplicate_aware):
                outcome = await asyncio.to_thread(self._remember, content, kind)
                if outcome.get("unsupported"):
                    # A wrapper can expose the method while its delegate has no
                    # judged write; fall back rather than fail the store.
                    outcome = await asyncio.to_thread(
                        self._remember_locally, content, kind
                    )
            else:
                outcome = await asyncio.to_thread(self._remember_locally, content, kind)
            if outcome.get("stored"):
                # The judged receipt is not evidence (X-04): the service has
                # answered merge/parent with an id a follow-up GET cannot find
                # (2026-10-29 probe
                # ~/Documents/xagent-memory-saas/.probe-live-20261029/y07-C9-merge.txt:
                # e8bcc505 / b46007db 404).
                # Verify before telling the model anything was combined.
                outcome = await asyncio.to_thread(
                    self._settle_receipt, outcome, content, kind
                )
            action = str(outcome.get("action") or "create")
            memory_id = outcome.get("memory_id")
            if outcome.get("stored"):
                # Account for the created resource before any further await so
                # trace delivery cannot reopen capacity after a cancellation.
                # A merged or attached write is still one stored fact.
                self.stored_count += 1

            if tracer is not None and task_id:
                await trace_memory_store_end(
                    tracer,
                    task_id,
                    data={
                        "storage_success": bool(outcome.get("stored")),
                        "memory_id": memory_id,
                        "storage_action": action,
                        "storage_verified": outcome.get("verified"),
                        "storage_degraded": outcome.get("degraded"),
                        "source": STORE_MEMORY_TOOL_NAME,
                        "step_id": step_id,
                    },
                )

            if not outcome.get("stored"):
                return {
                    "success": False,
                    "error": str(outcome.get("error") or "Failed to store memory."),
                }

            result: dict[str, Any] = {
                "success": True,
                "stored": True,
                "action": action,
                "memory_id": memory_id,
                **self._action_feedback(
                    action,
                    memory_id,
                    outcome.get("parent_id"),
                    verified=outcome.get("verified"),
                    recovered=bool(outcome.get("recovered")),
                    recovered_reason=outcome.get("recovery_reason"),
                    failed_receipt=outcome.get("failed_receipt"),
                    unverified_reason=outcome.get("unverified_reason"),
                    unverified_kind=outcome.get("unverified_kind"),
                ),
            }
            if outcome.get("verified") is not None:
                # X-04: report whether a read-back confirmed this write; a
                # store with no read path leaves the key out rather than
                # claiming a verification it cannot perform.
                result["verified"] = bool(outcome["verified"])
            if outcome.get("degraded") is not None:
                # The service's own degraded flag rides through instead of being
                # dropped, so the model can say the write landed partially
                # indexed rather than presenting it as fully settled.
                result["degraded"] = bool(outcome["degraded"])
            return result

    @staticmethod
    def _action_feedback(
        action: str,
        memory_id: Any,
        parent_id: Any,
        *,
        verified: Any = True,
        recovered: bool = False,
        recovered_reason: str | None = None,
        failed_receipt: Any = None,
        unverified_reason: str | None = None,
        unverified_kind: str | None = None,
    ) -> dict[str, Any]:
        """Tell the model which branch the judgement took (§10.4).

        Reusing an existing memory is a success, not a skip: the caller must
        see whether the new text replaced/merged an old note or was attached
        under a related one, so it does not report "skipped as duplicate".

        A merge/attach wording needs a live note behind the receipt (X-04): a
        receipt that could not be read back is only ever reported as the create
        that actually landed, never as "merged". A receipt whose note exists
        but does not carry the new text is not a merge either.
        """
        if verified is False:
            # The write may well have landed, so this is reported as
            # accepted-but-unconfirmed rather than failed, and the model is
            # told not to write it again (X-04).
            if unverified_reason == _NO_RECEIPT_ID_REASON:
                # No id means no read-back at all: say that, not "timed out".
                return {"message": MISSING_RECEIPT_ID_MESSAGE}
            if unverified_reason == _CONTENT_MISMATCH_REASON:
                # The write answered and its note is there; only the body
                # comparison failed, so no timeout/5xx wording either.
                return {
                    "message": (
                        RECOVERED_CONTENT_UNCONFIRMED_MESSAGE
                        if recovered
                        else CONTENT_UNCONFIRMED_MESSAGE
                    )
                }
            if recovered and recovered_reason == "content_missing":
                # A replacement note was created and answered; only its body
                # is unconfirmed. The timeout/5xx wording would be a lie.
                return {"message": RECOVERED_CONTENT_UNCONFIRMED_MESSAGE}
            # Third review: the cause decides the wording. A read-back the
            # service answered (401, another 4xx, an unparsable 2xx body) was
            # refused or is unreadable -- never "timed out". Only the transport
            # branch (timeout, connection failure, 5xx) may enumerate those
            # causes, and a failure that cannot be classified claims none.
            if unverified_kind == "refused":
                return {"message": READBACK_REJECTED_MESSAGE}
            if unverified_kind == "unreachable":
                return {"message": UNVERIFIED_RECEIPT_MESSAGE}
            return {"message": RECEIPT_FAILURE_UNKNOWN_MESSAGE}
        if recovered:
            if recovered_reason == "content_missing":
                return {
                    "message": (
                        f"The service reported {action} on {failed_receipt}, but "
                        "that note's body does not contain the new text, so it "
                        f"was stored as a new memory ({memory_id}) instead; "
                        "nothing was merged into an existing note."
                    )
                }
            return {
                "message": (
                    f"The service reported {action} on {failed_receipt or memory_id}, "
                    "but that memory cannot be read back, so it was stored as a "
                    "new memory instead; nothing was merged into an existing note."
                )
            }
        if action == "merge":
            if verified is False:
                # Unreachable by construction (an unverified merge is either
                # recovered or failed); kept so no path can claim a merge.
                return {"message": f"Stored as a new memory ({memory_id})."}
            if verified is None:
                return {
                    "message": (
                        "The store reported this write as merged into an "
                        f"existing similar memory ({memory_id}); this store "
                        "cannot read a note back, so the merge itself is "
                        "unconfirmed."
                    )
                }
            return {
                "message": (
                    "Merged into an existing similar memory "
                    f"({memory_id}); the new text was combined into that note "
                    "instead of creating a duplicate."
                )
            }
        if action == "parent":
            if verified is None:
                return {
                    "parent_id": parent_id,
                    "message": (
                        f"The store reported attaching this note as a child of "
                        f"a related memory ({parent_id}); the stored note is "
                        f"{memory_id}, and the parent link is unconfirmed."
                    ),
                }
            return {
                "parent_id": parent_id,
                "message": (
                    f"Attached as a child of a related memory ({parent_id}); "
                    f"the stored note is {memory_id}."
                ),
            }
        if action == "update":
            return {
                "message": (
                    f"Updated an existing similar memory in place ({memory_id})."
                )
            }
        return {
            "message": f"Stored as a new memory ({memory_id}).",
        }

    def _remember(self, content: str, kind: str) -> dict[str, Any]:
        """Judged write through a store that offers ``remember`` (§10.4).

        The store (the remote adapter) asks the service to pick merge /
        parent / create from the candidate similarity and reports the branch
        back, so one call replaces the old "search then skip" pair.
        """
        note = MemoryNote(
            content=content,
            category=self.category,
            metadata={
                "kind": kind,
                "source": STORE_MEMORY_TOOL_NAME,
            },
        )
        try:
            response = self.memory_store.remember(note)
        except (AttributeError, NotImplementedError):
            # The store advertises the method but does not support it (a
            # wrapper over a local store); the caller falls back.
            return {"unsupported": True}
        except Exception as exc:
            if _is_append_only_conflict(exc):
                # X-05: the service refuses a judged write that would modify a
                # delivered note ("delivered notes are append-only"). Asking it
                # for a merge again can only fail the same way, so the note is
                # created directly — no merge candidate, no parent — once.
                return self._create_after_conflict(content, kind, exc)
            logger.exception("store_memory judged write failed")
            # str() only: the adapter's message carries the service's repair
            # hint (the 409 append-only guidance), while the traceback stays
            # in the log instead of the model-facing text.
            return {
                "stored": False,
                "action": "failed",
                "error": str(exc) or "Failed to store memory.",
            }
        if not getattr(response, "success", False):
            return {
                "stored": False,
                "action": "failed",
                "error": str(
                    getattr(response, "error", None) or "Failed to store memory."
                ),
            }
        response_metadata = getattr(response, "metadata", None) or {}
        outcome = {
            "stored": True,
            "action": str(response_metadata.get("action") or "create"),
            "memory_id": getattr(response, "memory_id", None),
            "parent_id": response_metadata.get("parent_id"),
            "error": None,
        }
        if "degraded" in response_metadata:
            # Pass the upstream flag through untouched (True/False/None) so the
            # caller can report a partially applied write instead of a clean one.
            outcome["degraded"] = response_metadata.get("degraded")
        return outcome

    def _create_after_conflict(
        self, content: str, kind: str, conflict: BaseException
    ) -> dict[str, Any]:
        """Retry a 409 judged write as a plain create, exactly once (X-05).

        The judged path cannot choose another branch, so the note is written
        without a merge candidate or parent. A refused create fails loudly with
        both errors; a stored one reports ``action=create``, never a merge.
        """
        memory_id, add_error, add_degraded = self._add_with_error(content, kind)
        if not memory_id:
            return {
                "stored": False,
                "action": "failed",
                "error": (
                    f"{conflict} — retried as a new memory without a merge "
                    f"target, and that also failed: "
                    f"{add_error or 'no memory id returned'}. Nothing was saved."
                ),
            }
        result: dict[str, Any] = {
            "stored": True,
            "action": "create",
            "memory_id": memory_id,
            "parent_id": None,
            "error": None,
            "conflict": str(conflict),
        }
        if add_degraded is not None:
            # X-04: an upstream degraded flag on the create answer is carried
            # through instead of being lost on this branch.
            result["degraded"] = add_degraded
        return result

    def _settle_receipt(
        self, outcome: dict[str, Any], content: str, kind: str
    ) -> dict[str, Any]:
        """Confirm the judged write with one read-back, recovering if it is gone.

        A ``merge``/``parent`` receipt has been observed to name an id that a
        follow-up GET answers 404 for (2026-10-29 evidence
        ``~/Documents/xagent-memory-saas/.probe-live-20261029/y07-C9-merge.txt``:
        e8bcc505 and b46007db both 404 while the text
        survived under another id). The read-back uses the store's own read
        timeout, which the remote adapter already keeps short (2.0s), so the
        check never hangs the interactive path. A store without ``get`` is
        skipped rather than failed: there is nothing to verify against.

        The read-back is split into three states (X-04 rework). Only a note the
        service says does not exist (404/403/not-found) is re-created. A
        read-back that *could not be answered* — timeout, connection error,
        5xx, any other exception — says nothing about the write, so it is never
        retried: doing that wrote the same fact twice. It is reported as
        accepted-but-unverified instead. A merge/parent receipt whose note
        exists but whose body does not contain the new text was not merged: only
        then is the text re-created once as a new note (a live id alone is not
        evidence of a merge). A *create* receipt whose body does not carry the
        text is reported unverified instead — the note it names exists, so a
        second create would duplicate the same fact.
        """
        get = getattr(self.memory_store, "get", None)
        if not callable(get):
            return {**outcome, "verified": None}
        memory_id = outcome.get("memory_id")
        state, detail, note, unconfirmed_kind = _receipt_state(get, memory_id)

        if state == "unconfirmed":
            # The write may have landed; a failed read says nothing about it.
            return {
                **outcome,
                "verified": False,
                "unverified_reason": detail,
                "unverified_kind": unconfirmed_kind,
            }

        if state == "present":
            body = _receipt_body(note)
            if body is None:
                # The note exists but carries no readable body, so the merge
                # cannot be judged either way. Do not write a second copy.
                return {
                    **outcome,
                    "verified": False,
                    "unverified_reason": (
                        "the receipt note was returned without a readable body"
                    ),
                    # The service answered with a note that cannot be read:
                    # unreadable, not slow (third review).
                    "unverified_kind": "refused",
                }
            if _body_contains(body, content):
                return {**outcome, "verified": True}
            if str(outcome.get("action") or "create") not in ("merge", "parent"):
                # A create receipt is not a promise of a merge: the note it
                # names is a note of its own. If its body does not carry the
                # text, that is an unconfirmed write, not evidence for writing
                # the same fact a second time (X-04 rework).
                return {
                    **outcome,
                    "verified": False,
                    "unverified_reason": _CONTENT_MISMATCH_REASON,
                }
            failure_text = "the note behind the receipt does not contain the new text"
            recovery_reason = "content_missing"
        else:
            failure_text = f"that memory cannot be read back ({detail})"
            recovery_reason = "missing"

        # The receipt names no usable note: store the text as a new note so
        # the fact is not lost, and report the create that actually landed.
        new_id, add_error, add_degraded = self._add_with_error(content, kind)
        if not new_id:
            return {
                "stored": False,
                "action": "failed",
                "memory_id": memory_id,
                "verified": False,
                "error": (
                    f"The service reported {outcome.get('action')} on "
                    f"{memory_id}, but {failure_text}; re-storing it as a new "
                    f"memory failed too ({add_error or 'no memory id returned'}). "
                    "Nothing was saved."
                ),
            }
        new_state, new_detail, new_note, new_kind = _receipt_state(get, new_id)
        if new_state == "absent":
            return {
                "stored": False,
                "action": "failed",
                "memory_id": new_id,
                "verified": False,
                "error": (
                    f"The service reported {outcome.get('action')} on "
                    f"{memory_id} and the replacement note {new_id} cannot be "
                    f"read back either ({new_detail}). Nothing was saved."
                ),
            }
        verified = False
        mismatch = False
        if new_state == "present":
            new_body = _receipt_body(new_note)
            verified = new_body is not None and _body_contains(new_body, content)
            mismatch = not verified
        result: dict[str, Any] = {
            "stored": True,
            "action": "create",
            "memory_id": new_id,
            "parent_id": None,
            "error": None,
            "verified": verified,
            "recovered": True,
            "recovery_reason": recovery_reason,
            "failed_receipt": memory_id,
        }
        if not verified:
            # Distinguish "the replacement exists but its body differs" from
            # "the replacement could not be read back" so the message does not
            # claim a timeout that never happened (X-04 rework).
            result["unverified_reason"] = (
                _CONTENT_MISMATCH_REASON
                if mismatch
                else new_detail
                or ("the replacement note could not be read back with its body")
            )
            if not mismatch:
                # Third review: the replacement's read-back failed for a
                # specific cause; carry it so the message does not generalise
                # to a timeout.
                result["unverified_kind"] = new_kind or "unknown"
        if add_degraded is not None:
            result["degraded"] = add_degraded
        return result

    def _remember_locally(self, content: str, kind: str) -> dict[str, Any]:
        """Duplicate-aware fallback for stores without a judged write.

        The local LanceDB store has no candidate scoring and no parent
        relation on a note, so a similar hit is folded into the existing note
        (an in-place update) rather than dropped: the memory is never lost,
        which is the point of the change.
        """
        existing = self._find_similar(content)
        if existing is not None:
            updated = self._merge_existing(existing, content)
            if updated:
                return {
                    "stored": True,
                    "action": "update",
                    "memory_id": getattr(existing, "id", None),
                    "parent_id": None,
                    "error": None,
                }
        memory_id = self._add(content, kind)
        if not memory_id:
            return {
                "stored": False,
                "action": "failed",
                "error": "Failed to store memory.",
            }
        return {
            "stored": True,
            "action": "create",
            "memory_id": memory_id,
            "parent_id": None,
            "error": None,
        }

    def _merge_existing(self, existing: Any, content: str) -> bool:
        """Fold new text into an existing note, keeping the visible history.

        X-06 tag cleanup for a merge, deliberately conservative: the note is
        rewritten as ``old body + new text``, so the merged body still asserts
        everything the old body said. Charging ``drop_replaced_tags`` with the
        incoming fragment alone cleared tags the merged note still supports
        (a classified value cited only in the kept history). The merged body is
        therefore the "new content", which means nothing is dropped unless the
        old text really disappeared — when in doubt, fewer tags are cleared,
        never a live classification tag (probed 2026-10-29,
        ``~/Documents/xagent-memory-saas/.probe-x06-tags-20261029/``).
        """
        try:
            old_content = str(getattr(existing, "content", "") or "").strip()
            merged_content = (
                old_content
                if content in old_content
                else (f"{old_content}\n{content}" if old_content else content)
            )
            if merged_content != old_content:
                # X-06: the rewritten note must not keep a tag that only the
                # superseded statement supported. The appended body keeps the
                # old text as history, so the comparison body is the merged
                # text, not the fragment (see the docstring trade-off above).
                previous_body = existing.content
                existing.content = merged_content
                drop_replaced_tags(existing, previous_body)
            response = self.memory_store.update(existing)
        except Exception:
            logger.exception("store_memory failed to merge into a similar memory")
            return False
        return bool(getattr(response, "success", False))

    def _find_similar(self, content: str) -> Any | None:
        search = getattr(self.memory_store, "search", None)
        if not callable(search):
            return None
        try:
            results = search(
                query=content,
                k=1,
                filters={"category": self.category},
                similarity_threshold=self.dedup_distance_threshold,
            )
        except Exception:
            logger.exception("store_memory dedup search failed; storing anyway")
            return None
        return results[0] if results else None

    def _add(self, content: str, kind: str) -> str | None:
        memory_id, _, _ = self._add_with_error(content, kind)
        return memory_id

    def _add_with_error(
        self, content: str, kind: str
    ) -> tuple[str | None, str | None, Any]:
        """Create a note directly and return ``(memory_id, error, degraded)``.

        The error text is kept so a failed recovery can say why the plain
        create did not land instead of only reporting that it did not. The
        third value is the store's own ``degraded`` flag when the create answer
        carries one (X-04): the remote adapter's ``add`` returns a bare
        ``MemoryResponse`` without a ``degraded`` key today, so nothing is
        invented — the flag is forwarded only when a store actually reports it.
        """
        note = MemoryNote(
            content=content,
            category=self.category,
            # No request text here: it is conversation content, and a note
            # outlives the conversation it came from (#2753).
            metadata={
                "kind": kind,
                "source": STORE_MEMORY_TOOL_NAME,
            },
        )
        try:
            response = self.memory_store.add(note)
        except Exception as exc:
            logger.exception("store_memory failed to add memory")
            return None, str(exc) or None, None
        if not getattr(response, "success", False):
            return (
                None,
                str(getattr(response, "error", None) or "add refused"),
                None,
            )
        metadata = getattr(response, "metadata", None)
        degraded = (
            metadata.get("degraded")
            if isinstance(metadata, dict) and "degraded" in metadata
            else None
        )
        return getattr(response, "memory_id", None), None, degraded


def _is_append_only_conflict(exc: BaseException) -> bool:
    """Whether the failure is the service's 409 "append-only" conflict (X-05).

    The remote adapter raises :class:`RemoteMemoryStoreError` with the HTTP
    status attached and the upstream message embedded ("Remote memory service
    returned HTTP 409（code=conflict）… delivered notes are append-only"), so
    both the status attribute and the text are checked; a fake store in tests
    only carries the text.
    """
    if getattr(exc, "status_code", None) == 409:
        return True
    text = str(exc).lower()
    return "http 409" in text or "append-only" in text or "append only" in text


#: Statuses that positively answer "no such note": ``remote.py::get`` maps both
#: 403 and 404 to ``NOT_FOUND_OR_DENIED_ERROR``, so from the caller's point of
#: view the note is not there and may be written again. The numeric statuses
#: are the only trusted signal on an exception: a UUID or an upstream message
#: can contain "403"/"not found" without meaning anything.
_ABSENT_STATUS_CODES = (403, 404)
#: The only text accepted as an absence verdict when no status is attached: the
#: service's own not-found constant, matched on the whole message (see
#: ``_is_not_found_text``), never as a substring -- the 401 classifier embeds
#: the upstream ``message`` ("identity not found", "access denied") in its text
#: (``remote.py::_classify_401``), and a substring match would turn a credential
#: failure into a re-create.
_ABSENT_TEXT = " ".join(NOT_FOUND_OR_DENIED_ERROR.split()).casefold()
#: Reason reported when a receipt carries no id (X-04 rework): the write is
#: unconfirmed, not absent, and gets its own message.
_NO_RECEIPT_ID_REASON = "the receipt carried no memory id"
#: Reason reported when a create receipt's note exists but its body does not
#: carry the new text: also unconfirmed, with its own message.
_CONTENT_MISMATCH_REASON = "the created note's body does not contain the new text"
#: Read-back failures where the service *answered* even though it raised, and
#: the body could not be read as a note. ``remote.py`` raises these as a plain
#: ``RemoteMemoryStoreError`` with no status (``_json`` / ``get``).
_READBACK_UNREADABLE_MARKERS = ("non-json response", "unexpected note")
#: Transport failures ``remote.py`` reports with no status: both messages start
#: with "Remote memory upstream unreachable" and then name the transport cause.
_READBACK_TRANSPORT_MARKERS = (
    "upstream unreachable",
    "timed out",
    "timeout",
    "connection failure",
    "connection error",
    "connection reset",
)


def _unconfirmed_readback_kind(exc: BaseException | None, detail: str) -> str:
    """Place an unconfirmed read-back in ``refused``/``unreachable``/``unknown``.

    The status is trusted over the text: a 5xx is a server-side failure that
    may or may not have processed the request, while any other answered 4xx
    (403/404 are already an ``absent`` verdict by the time this runs) means the
    service refused the read. Without a status, ``remote.py``'s two 2xx body
    failures read as unparsable content and its transport failures name the
    transport; anything else claims no cause at all.
    """
    status = getattr(exc, "status_code", None) if exc is not None else None
    if isinstance(status, int):
        if 500 <= status <= 599:
            return "unreachable"
        if 400 <= status <= 499:
            return "refused"
    text = (detail or "").casefold()
    if any(marker in text for marker in _READBACK_UNREADABLE_MARKERS):
        return "refused"
    if any(marker in text for marker in _READBACK_TRANSPORT_MARKERS):
        return "unreachable"
    return "unknown"


def _is_not_found_text(text: Any) -> bool:
    """Whether the whole error text is the service's not-found constant.

    ``str.startswith`` after collapsing whitespace and casefolding: the adapter
    returns the constant alone (``_not_found_or_denied``), while a test fake or
    an older adapter may append a path. Nothing else counts -- in particular a
    text that merely *contains* "not found" is a different failure.
    """
    candidate = " ".join(str(text or "").split()).casefold()
    return bool(candidate) and candidate.startswith(_ABSENT_TEXT)


def _receipt_state(get: Any, memory_id: Any) -> tuple[str, str | None, Any, str | None]:
    """Classify one read-back as ``absent`` / ``present`` / ``unconfirmed``.

    ``remote.py::get`` answers 403/404 with ``success=False`` and
    ``NOT_FOUND_OR_DENIED_ERROR`` instead of raising; a read timeout, a
    connection failure, a 5xx and a non-JSON body all raise
    ``RemoteMemoryStoreError`` (the 5xx and the 401 carrying ``status_code``).
    The distinction is the whole point of the X-04 rework: only ``absent`` is
    evidence that the write did not land.

    Absence needs positive evidence: a 403/404 status, or the not-found
    constant as the whole message. A receipt without an id is *unconfirmed* —
    the service may have stored the note and simply not reported the id, so
    creating again could duplicate it.

    The fourth element classifies an unconfirmed read-back (``refused`` /
    ``unreachable`` / ``unknown``) so the caller can word the failure by its
    actual cause instead of claiming a timeout for every one of them.
    """
    if not memory_id:
        return "unconfirmed", _NO_RECEIPT_ID_REASON, None, None
    try:
        response = get(str(memory_id))
    except Exception as exc:
        detail = str(exc) or "the read-back failed"
        if getattr(exc, "status_code", None) in _ABSENT_STATUS_CODES:
            return "absent", detail, None, None
        if _is_not_found_text(detail):
            return "absent", detail, None, None
        return "unconfirmed", detail, None, _unconfirmed_readback_kind(exc, detail)
    if not getattr(response, "success", False):
        detail = str(getattr(response, "error", None) or "the read-back failed")
        if _is_not_found_text(detail):
            return "absent", detail, None, None
        return "unconfirmed", detail, None, "unknown"
    return "present", None, getattr(response, "content", None), None


def _receipt_body(note: Any) -> str | None:
    """Return the note's body text, or ``None`` when it cannot be read."""
    if note is None:
        return None
    value = note if isinstance(note, (str, bytes)) else getattr(note, "content", None)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _normalized_receipt_text(value: Any) -> str:
    """Case- and whitespace-insensitive body text for the containment check.

    X-04 rework: the service may normalise case or reflow the body, so the
    check folds case (``casefold``, the same reduction ``core.py`` uses) and
    collapses every whitespace run to one space instead of comparing bytes.
    """
    return " ".join(str(value or "").split()).casefold()


def _body_contains(body: str, content: str) -> bool:
    """Whether ``body`` holds ``content`` once case and whitespace are folded."""
    wanted = _normalized_receipt_text(content)
    if not wanted:
        return False
    return wanted in _normalized_receipt_text(body)


#: Accepted values of the optional ``space`` search filter (方案 §4.7): the
#: personal notes, the department layer, or the whole organisation. The server
#: resolves the caller's readable layers itself, so ``space`` only narrows.
SearchSpace = Literal["personal", "department", "company"]
_SEARCH_SPACE_VALUES = ("personal", "department", "company")


def _search_filters(space: Any, category: Any) -> tuple[dict[str, Any], str | None]:
    """Build the optional ``space``/``category`` filter keys (§4.7).

    Returns ``(filters, None)`` for valid input and ``({}, message)`` for an
    illegal ``space``: the tool must tell the model which values exist instead
    of silently searching the unfiltered surface.
    """
    filters: dict[str, Any] = {}
    if space:
        value = str(space)
        if value not in _SEARCH_SPACE_VALUES:
            return (
                {},
                "space must be one of "
                + ", ".join(_SEARCH_SPACE_VALUES)
                + f" (got {value!r}); omit it to search everything.",
            )
        filters["space"] = value
    if category:
        filters["category"] = str(category)
    return filters, None


class SearchMemoryArgs(BaseModel):
    query: str = Field(
        description=(
            "What to look for in stored memories, written as one "
            "complete sentence — retrieval is semantic and matches "
            "meaning, not exact keywords."
        )
    )
    limit: int = Field(
        default=DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description="Maximum number of memories to return.",
    )
    space: SearchSpace | None = Field(
        default=None,
        description=(
            "Optional shared-space narrowing: 'personal' for your own notes, "
            "'department' for your department's shared layer, 'company' for "
            "the whole organisation. Omit to search everything you can read."
        ),
    )
    category: str | None = Field(
        default=None,
        description="Optional exact category filter (for example react_memory).",
    )


class SearchMemoryTool:
    """Execution-scoped ``search_memory`` tool bound to a memory store."""

    name = SEARCH_MEMORY_TOOL_NAME
    description = SEARCH_MEMORY_DESCRIPTION
    args_schema = SearchMemoryArgs

    def __init__(self, *, memory_store: Any, runtime: Any | None = None) -> None:
        self.memory_store = memory_store
        self.runtime = runtime

    async def execute(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        space: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            return {"success": False, "error": "query must be a non-empty string."}
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = DEFAULT_SEARCH_LIMIT
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))
        filters, filter_error = _search_filters(space, category)
        if filter_error is not None:
            return {"success": False, "error": filter_error}

        task_id = str(_runtime_attr(self.runtime, "execution_id") or "")
        step_id = _runtime_attr(self.runtime, "active_react_step_id")
        tracer = _runtime_attr(self.runtime, "tracer")

        if tracer is not None and task_id:
            await trace_memory_retrieve_start(
                tracer,
                task_id=task_id,
                step_id=step_id,
                data={"query": query[:200], "source": SEARCH_MEMORY_TOOL_NAME},
            )

        error: str | None = None
        try:
            memories = await asyncio.to_thread(self._search, query, limit, filters)
        except Exception as exc:
            logger.exception("search_memory failed")
            memories = []
            # surface the service's own message (§4.7 errors reach the model);
            # str() carries no traceback.
            error = str(exc) or "Failed to search memories."

        if tracer is not None and task_id:
            await trace_memory_retrieve_end(
                tracer,
                task_id=task_id,
                step_id=step_id,
                data={
                    "query": query[:200],
                    "memories_count": len(memories),
                    "found": bool(memories),
                    "success": error is None,
                    "source": SEARCH_MEMORY_TOOL_NAME,
                },
            )

        if error is not None:
            return {"success": False, "error": error}
        return {"success": True, "memories": memories, "count": len(memories)}

    def _search(
        self, query: str, limit: int, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # Only the caller's own keys ride here; the execution-scope keys are
        # merged downstream (UserIsolatedMemoryStore / remote._filters_payload),
        # so passing the dict through merges instead of overwriting them.
        results = self.memory_store.search(
            query=query,
            k=limit,
            filters=dict(filters) if filters else None,
        )
        memories = []
        for note in results or []:
            content = getattr(note, "content", "") or ""
            space_kind = getattr(note, "space_kind", None)
            space_id = getattr(note, "space_id", None)
            if space_kind:
                # provenance rides with the quote: the model must know it is citing the
                # team's shared layer, not somebody's private note (§A4 hand-over form)
                tag = f"[shared:{space_kind}" + (f":{space_id}]" if space_id else "]")
                content = f"{tag} {content}"
            entry: dict[str, Any] = {
                "id": getattr(note, "id", None),
                "content": content,
                "category": getattr(note, "category", "general"),
                "space_kind": space_kind,
                "space_id": space_id,
            }
            # X-07: an organisation-scope search mixes several people's notes.
            # The server already hides everything a regular user may not read,
            # so this is not an access widening -- it is the attribution the
            # answer needs to say whose fact it is. Only the stored owner is
            # reported; a note without one stays without one.
            note_metadata = getattr(note, "metadata", None) or {}
            owner_user_id = (
                note_metadata.get("user_id")
                if isinstance(note_metadata, dict)
                else None
            )
            if owner_user_id not in (None, ""):
                entry["owner_user_id"] = owner_user_id
            memories.append(entry)
        return memories


class UpdateMemoryArgs(BaseModel):
    memory_id: str = Field(
        description="Id of the memory to update, as returned by search_memory."
    )
    content: str = Field(
        description="Corrected, self-contained memory text that replaces the "
        "old content."
    )


class UpdateMemoryTool:
    """Execution-scoped ``update_memory`` tool bound to a memory store."""

    name = UPDATE_MEMORY_TOOL_NAME
    description = UPDATE_MEMORY_DESCRIPTION
    args_schema = UpdateMemoryArgs

    def __init__(self, *, memory_store: Any) -> None:
        self.memory_store = memory_store

    async def execute(self, memory_id: str, content: str) -> dict[str, Any]:
        memory_id = str(memory_id or "").strip()
        content = str(content or "").strip()
        if not memory_id or not content:
            return {
                "success": False,
                "error": "memory_id and content must be non-empty strings.",
            }
        return await asyncio.to_thread(self._update, memory_id, content)

    def _update(self, memory_id: str, content: str) -> dict[str, Any]:
        try:
            existing = self.memory_store.get(memory_id)
            if not getattr(existing, "success", False):
                return {
                    "success": False,
                    "error": (
                        f"Memory '{memory_id}' not found or not accessible (it may "
                        "exist under another owner). Do not tell the user it was "
                        "never recorded — search first, then ask whether the id is "
                        "right."
                    ),
                }
            note = getattr(existing, "content", None)
            if note is None:
                return {
                    "success": False,
                    "error": f"Memory '{memory_id}' has no stored note to update.",
                }
            previous_content = note.content
            note.content = content
            # X-06: the update tool never sets tags, and the adapter sends the
            # note's tag list verbatim, so a tag that only the replaced text
            # supported would survive the rewrite (京A8888 kept under the new
            # 京A9999). Drop those before writing; a classification tag that
            # never appeared in the body is left alone.
            if previous_content != content:
                drop_replaced_tags(note, previous_content)
            if note.metadata is None:
                note.metadata = {}
            response = self.memory_store.update(note)
        except Exception as exc:
            logger.exception("update_memory failed")
            # str() only — the service's 409 message ("send a follow-up
            # delivery instead") is the repair the model needs here.
            return {"success": False, "error": str(exc) or "Failed to update memory."}
        if not getattr(response, "success", False):
            return {
                "success": False,
                "error": str(getattr(response, "error", None) or "Update failed."),
            }
        return {"success": True, "memory_id": memory_id}


class DeleteMemoryArgs(BaseModel):
    memory_id: str = Field(
        description="Id of the memory to delete, as returned by search_memory."
    )


class DeleteMemoryTool:
    """Execution-scoped ``delete_memory`` tool bound to a memory store."""

    name = DELETE_MEMORY_TOOL_NAME
    description = DELETE_MEMORY_DESCRIPTION
    args_schema = DeleteMemoryArgs

    def __init__(self, *, memory_store: Any) -> None:
        self.memory_store = memory_store

    async def execute(self, memory_id: str) -> dict[str, Any]:
        memory_id = str(memory_id or "").strip()
        if not memory_id:
            return {"success": False, "error": "memory_id must be a non-empty string."}
        return await asyncio.to_thread(self._delete, memory_id)

    def _delete(self, memory_id: str) -> dict[str, Any]:
        try:
            response = self.memory_store.delete(memory_id)
        except Exception as exc:
            logger.exception("delete_memory failed")
            return {"success": False, "error": str(exc) or "Failed to delete memory."}
        if not getattr(response, "success", False):
            return {
                "success": False,
                "error": str(getattr(response, "error", None) or "Delete failed."),
            }
        return {"success": True, "memory_id": memory_id}


class DeliverMemoryArgs(BaseModel):
    target: str = Field(
        description=(
            "Who receives it: a teammate's username, external id or agent "
            "name, for example t10-probe."
        )
    )
    content: str = Field(
        description=(
            "The note to hand over — one self-contained statement, the same "
            "shape as store_memory content."
        )
    )
    attach: list[str] = Field(
        default_factory=list,
        description=(
            "Memories to ride along as quoted attachments: memory ids when "
            "you have them, or short keywords to look them up. The quoted "
            "text is appended to the delivery so the recipient can read it "
            "without a second lookup — treat it like quoting a document in a "
            "message."
        ),
    )


class DeliverMemoryTool:
    """Execution-scoped ``deliver_memory`` tool bound to a memory store."""

    name = DELIVER_MEMORY_TOOL_NAME
    description = DELIVER_MEMORY_DESCRIPTION
    args_schema = DeliverMemoryArgs

    def __init__(self, *, memory_store: Any) -> None:
        self.memory_store = memory_store

    async def execute(
        self, target: str, content: str, attach: list[str] | None = None
    ) -> dict[str, Any]:
        target = str(target or "").strip()
        content = str(content or "").strip()
        if not target:
            return {
                "success": False,
                "error": "target must be a non-empty recipient id.",
            }
        if not content:
            return {
                "success": False,
                "error": "content must be a non-empty statement.",
            }
        # production hands this tool a wrapper chain (revocation / user isolation)
        # over the real adapter — a class-name check would always miss it, the
        # same trap that broke the seat override (bounded unwrap instead).
        if _unwrap_remote_store(self.memory_store) is None:
            return {
                "success": False,
                "error": (
                    "delivery needs the remote memory service; the active "
                    "store has no teammate inbox."
                ),
            }
        quoted, unresolved = await self._quote_attachments(attach or [])
        if quoted:
            content = f"{content}\n\n— attached memories —\n{quoted}"
        note = MemoryNote(content=f"[to:{target}] {content}")
        try:
            response = await asyncio.to_thread(self.memory_store.add, note)
        except Exception as exc:
            logger.exception("deliver_memory failed")
            # the service's own message (rate limit, unknown target, 409) is
            # the only guidance the model can act on; str() excludes traceback.
            return {
                "success": False,
                "error": str(exc) or "Failed to deliver the note.",
            }
        if not getattr(response, "success", False):
            return {
                "success": False,
                "error": str(getattr(response, "error", None) or "Delivery failed."),
            }
        result = {
            "success": True,
            "memory_id": getattr(response, "memory_id", None),
            "delivered_to": target,
            "attachments_quoted": (len(attach or []) - len(unresolved)),
        }
        if unresolved:
            # the sender must SEE the stale refs: silently swapping in whatever was
            # nearest turned a broken quote into a confident wrong one
            result["attachments_unresolved"] = unresolved
        return result

    async def _quote_attachments(self, refs: list[str]) -> tuple[str, list[str]]:
        """Resolve each attachment ref into quoted memory text.

        A ref that looks like an id is fetched directly and never fuzzy-matched — an id
        that no longer exists (or that the sender may not read) must surface as
        unresolved, not resolve to whatever note happens to be nearest. Wording refs
        search with a relevance ceiling so a near-miss comes back unresolved too.
        Unresolvable refs are skipped rather than failing the delivery — a message with
        a stale quote still carries its own text.
        """
        lines: list[str] = []
        seen: set[str] = set()
        unresolved: list[str] = []
        for ref in list(refs)[:8]:
            ref = str(ref or "").strip()
            if not ref or ref in seen:
                continue
            seen.add(ref)
            note = None
            if len(ref) >= 32 and "-" in ref:
                get = getattr(self.memory_store, "get", None)
                if not callable(get):
                    unresolved.append(ref)
                    continue
                try:
                    response = await asyncio.to_thread(get, ref)
                    if getattr(response, "success", False):
                        candidate = getattr(response, "content", None)
                        if candidate is not None:
                            note = candidate
                except Exception:
                    logger.debug("attachment id lookup failed", exc_info=True)
                if note is None:
                    # an id ref is either gone or invisible: both mean "quote not found"
                    unresolved.append(ref)
                    continue
            if note is None:
                search = getattr(self.memory_store, "search", None)
                if callable(search):
                    try:
                        results = await asyncio.to_thread(
                            lambda: search(
                                query=ref,
                                k=1,
                                similarity_threshold=ATTACH_WORDING_MAX_DISTANCE,
                            )
                        )
                        note = results[0] if results else None
                    except Exception:
                        logger.debug("attachment keyword lookup failed", exc_info=True)
            if note is None:
                unresolved.append(ref)
                continue
            text = str(getattr(note, "content", "") or "").strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines), unresolved


class CheckInboxArgs(BaseModel):
    limit: int = Field(
        default=20,
        description="How many of the most recent deliveries to return (1-50).",
    )


class CheckInboxTool:
    """Execution-scoped ``check_inbox`` tool bound to a memory store."""

    name = CHECK_INBOX_TOOL_NAME
    description = CHECK_INBOX_DESCRIPTION
    args_schema = CheckInboxArgs

    def __init__(self, *, memory_store: Any) -> None:
        self.memory_store = memory_store

    def _inbox_surface(self) -> tuple[Any | None, Any | None]:
        """Resolve the dedicated inbox surface behind the wrapper chain.

        Probe order: the unwrapped ``RemoteMemoryStore`` first — production
        publishes a wrapper chain (revocation / user isolation) that can hide
        optional capabilities from a plain ``getattr`` on the outer object —
        then the store this tool was handed, so a bare local/fake store that
        defines ``inbox_list`` itself keeps working.

        Whichever probe finds the capability, the call binds to the outer
        store when it also forwards the method, so a revoked reference still
        fails closed through ``_live()`` instead of reaching around the
        wrapper.
        """
        remote = _unwrap_remote_store(self.memory_store)
        for probe in (remote, self.memory_store):
            if probe is None or not callable(getattr(probe, "inbox_list", None)):
                continue
            target = probe
            if probe is not self.memory_store and callable(
                getattr(self.memory_store, "inbox_list", None)
            ):
                target = self.memory_store
            return (
                getattr(target, "inbox_list", None),
                getattr(target, "inbox_read", None),
            )
        return None, None

    async def execute(self, limit: int = 20) -> dict[str, Any]:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, 50))
        # Prefer the dedicated inbox surface (unread-first + receipts beside the
        # immutable delivery). Legacy/fake stores without it keep the list+filter path.
        inbox_list, inbox_read = self._inbox_surface()
        if callable(inbox_list):
            try:
                notes = await asyncio.to_thread(lambda: inbox_list(include_read=False))
            except Exception as exc:
                if not _inbox_surface_missing(exc):
                    logger.exception("check_inbox failed")
                    return {
                        "success": False,
                        "error": str(exc) or "Failed to read the inbox.",
                    }
                # An old service without the inbox routes (404) or a wrapper
                # whose delegate lacks the capability (AttributeError): fall
                # back to the legacy list path instead of failing the tool.
                logger.warning(
                    "check_inbox found no usable inbox surface (%s); "
                    "falling back to list_all",
                    exc,
                )
            else:
                return await self._surface_result(notes, inbox_read, limit)
        return await self._legacy_result(limit)

    async def _surface_result(
        self, notes: list[Any], inbox_read: Any, limit: int
    ) -> dict[str, Any]:
        ordered = sorted(
            notes, key=lambda n: str(getattr(n, "timestamp", "") or ""), reverse=True
        )
        shown = ordered[:limit]
        items = []
        for note in shown:
            metadata = getattr(note, "metadata", None) or {}
            items.append(
                {
                    "id": getattr(note, "id", None),
                    "content": str(getattr(note, "content", "") or ""),
                    "from_user": metadata.get("from_user"),
                    "timestamp": str(getattr(note, "timestamp", "") or ""),
                }
            )
        marked = 0
        ids = [i["id"] for i in items if i.get("id")]
        if ids and callable(inbox_read):
            try:
                marked = await asyncio.to_thread(lambda: inbox_read(ids))
            except Exception:
                # receipts are best effort: the deliveries were still returned
                logger.warning("check_inbox could not take receipts", exc_info=True)
        return {
            "success": True,
            "count": len(items),
            "deliveries": items,
            "marked_read": marked,
            "unread_remaining": max(0, len(ordered) - len(shown)),
        }

    async def _legacy_result(self, limit: int) -> dict[str, Any]:
        list_all = getattr(self.memory_store, "list_all", None)
        if not callable(list_all):
            return {
                "success": False,
                "error": "the active memory store cannot list an inbox.",
            }
        try:
            notes = await asyncio.to_thread(list_all, {"metadata": {"deliver": True}})
        except Exception as exc:
            logger.exception("check_inbox failed")
            return {
                "success": False,
                "error": str(exc) or "Failed to read the inbox.",
            }
        ordered = sorted(
            notes, key=lambda n: str(getattr(n, "timestamp", "") or ""), reverse=True
        )
        items = []
        for note in ordered[:limit]:
            metadata = getattr(note, "metadata", None) or {}
            items.append(
                {
                    "id": getattr(note, "id", None),
                    "content": str(getattr(note, "content", "") or ""),
                    "from_user": metadata.get("from_user"),
                    "timestamp": str(getattr(note, "timestamp", "") or ""),
                }
            )
        return {"success": True, "count": len(items), "deliveries": items}


def _inbox_surface_missing(exc: BaseException) -> bool:
    """Whether the failure means "this store has no dedicated inbox surface".

    An old service answers 404 for the inbox side-table routes, and a wrapper
    whose delegate lacks the capability raises ``AttributeError`` on the call —
    both must fall back to the legacy ``list_all`` path. Everything else
    (5xx, 401/403, timeout, a revoked publication) is a real failure and
    surfaces to the caller.
    """
    if isinstance(exc, AttributeError):
        return True
    if getattr(exc, "status_code", None) == 404:
        return True
    return isinstance(exc, RemoteMemoryStoreError) and "HTTP 404" in str(exc)


def build_store_memory_tool(
    *,
    memory_store: Any | None,
    task: str,
    runtime: Any | None = None,
    category: str = "react_memory",
) -> StoreMemoryTool | None:
    """Create a ``store_memory`` tool, or None when no memory store is active."""

    if memory_store is None:
        return None
    return StoreMemoryTool(
        memory_store=memory_store,
        task=task,
        runtime=runtime,
        category=category,
    )


def build_memory_tools(
    *,
    memory_store: Any | None,
    task: str,
    runtime: Any | None = None,
    context: Any | None = None,
) -> list[Any]:
    """Create the memory tool set, or [] when no memory store is active.

    When ``context`` is given, mark it so the system context renders the
    memory-usage guidance alongside the tools (see
    ``MEMORY_TOOLS_METADATA_KEY``).
    """

    if memory_store is None:
        return []
    tools: list[Any] = [
        StoreMemoryTool(memory_store=memory_store, task=task, runtime=runtime),
        SearchMemoryTool(memory_store=memory_store, runtime=runtime),
        UpdateMemoryTool(memory_store=memory_store),
        DeleteMemoryTool(memory_store=memory_store),
        DeliverMemoryTool(memory_store=memory_store),
        CheckInboxTool(memory_store=memory_store),
    ]
    if tools and context is not None and hasattr(context, "metadata"):
        context.metadata[MEMORY_TOOLS_METADATA_KEY] = True
    return tools


def _runtime_attr(runtime: Any | None, name: str) -> Any | None:
    if runtime is None:
        return None
    return getattr(runtime, name, None)
