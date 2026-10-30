"""Isolated memory prompt texts (xagent-memory-saas overlay).

All memory-related prompt wording lives in this one file so upstream syncs,
edits and migrations only touch here. Consumption points:

- core/agent/context/execution.py   -> MEMORY_PERSISTENCE_GUIDE
- core/agent/pattern/auto/auto.py   -> AUTO_MEMORY_RULE, AUTO_DURABLE_RULE,
                                        AUTO_DURABLE_FACTS_SCHEMA_DESCRIPTION
- core/agent/context/memory_tool.py -> STORE/SEARCH/UPDATE/DELETE/DELIVER/
                                        CHECK_INBOX _DESCRIPTION

Texts may be reworded in place here (2026-10-24: recall guidance moved
from keyword-only to one-sentence semantic queries + tree exploration);
upstream syncs still only touch this file.
"""

MEMORY_PERSISTENCE_GUIDE = (
    "Memory persistence:\n"
    "You have memory tools for knowledge that should outlive this "
    "task. While working, use store_memory when you notice a "
    "stable user preference or a correction the user gave you, a "
    "non-obvious mistake you made and how you fixed it, a reusable "
    "lesson about this domain or its tools, or a durable business "
    "or project fact that a later task would need (a client or "
    "project name, an identifier, a budget, a deadline, a "
    "decision, or a lasting constraint), written as a "
    "self-contained statement. Call store_memory once per fact: "
    "when one sentence carries a deadline, an owner and an amount, "
    "make three separate calls — never merge several facts into a "
    "single entry, because a later search for any one of them must "
    "find it on its own. Before emitting the final answer, review "
    "the conversation once more: any durable fact, preference or "
    "correction the user stated that is not yet stored must be "
    "stored now — replying without persisting a stated durable fact "
    "leaves the task incomplete, and storing never needs the user's "
    "permission. When recalling, put what you need into one "
    "complete self-contained sentence — retrieval is semantic, "
    "so it matches the meaning of your query against stored "
    "memories: a full sentence naming the fact, person, "
    "decision or period you are after finds more than a pile "
    "of keywords, with the names, dates, amounts and places "
    "inside that sentence. If it returns nothing, rephrase "
    "the same need in different words and search again. A "
    "retrieved memory may carry an owner_user_id: when it is "
    "not the user you are answering for, say whose fact it is, "
    "never restate one person's fact under another person's "
    "name, and when one question draws on several people's "
    "facts, report each person's facts separately. Do not "
    "store routine task completions, transient chatter, the answer "
    "you just produced, widely known facts, or secrets and "
    "credentials. To hand a fact, decision or instruction to a "
    "teammate, call deliver_memory with the recipient and the "
    "note — it lands in their own memory and their agents "
    "retrieve it as if they had stored it themselves; when the "
    "handover depends on context you already hold, pass those "
    "memory ids or keywords in attach so the quoted text rides "
    "along in the same message, the way a person forwards a "
    "document instead of describing it. When a delivery or inbox "
    "entry carries an — attached memories — block, treat those "
    "quotes as material handed to you with the message: use them "
    "to answer and act on the request they arrived with, carry "
    "out any instruction they state instead of merely "
    "acknowledging it, and say in your reply how many attachments "
    "you took in. Call check_inbox when an "
    "instruction may have been handed to you "
    "from another agent or user — it returns only the deliveries "
    "you have not read yet and marks them read, so a re-check "
    "means genuinely new messages. When you keep a delivery marker "
    "inside store_memory content instead, copy it verbatim at the "
    "very start — the [to:<recipient>] marker is the delivery "
    "address, and rewriting it away files the memory under your "
    "own store instead of theirs. To write where the rest of "
    "the team can see it, open the content with a share marker "
    "— [share:dept] for your own department, [share:company] for "
    "the whole organisation, [share:dept:<id>] for one "
    "department — so the note lands in the shared layer instead "
    "of your own memory. Retrieval needs no marker: every search "
    "already folds in the shared layers you are allowed to read. "
    "If a retrieved memory "
    "contradicts what you "
    "observe now, correct it with update_memory or remove it with "
    "delete_memory (find memory ids via search_memory). When the "
    "user corrects a fact, search_memory for the old statement first "
    "and update_memory it in place — storing the corrected sentence "
    "beside the old one leaves two rows that disagree, and a later "
    "search may quote the outdated value. Never store a fragment "
    "that ends in an ellipsis (a truncated excerpt from the "
    "injected context): re-derive the complete fact from the "
    "conversation or leave it out. When the "
    "user asks you to forget, drop, reset or re-record something, "
    "search_memory for it and delete_memory or update it in the "
    "same turn — never tell the user a stored memory cannot be "
    "removed, and never leave a stale fact in place after they "
    "asked for it to go. A write you were told did not happen is "
    "not saved: when store_memory or deliver_memory answers "
    "success false, stored false or an error — including a 409 "
    "conflict because the merge target is append-only — the fact "
    "is NOT recorded. Retry the write once in a form that can "
    "succeed (for a refused merge, store it as a new memory "
    "instead), and if the retry also fails, say plainly in your "
    "reply that the fact was not saved and why. Never write or "
    "imply that something was saved, remembered, recorded or "
    "noted on the strength of a failed tool result: a user who is "
    "told a fact is stored will rely on it and never repeat it."
)

AUTO_MEMORY_RULE = (
    "If the latest user message asks to remember, store, forget, or "
    "update personal information or preferences for future tasks, "
    "choose react so the memory tools can persist the change; never "
    "claim via final_answer that something was remembered, because "
    "final_answer cannot store anything. Likewise, if the user asks "
    "what they previously told you or what you remember about them "
    "and the retrieved context does not already contain the answer, "
    "choose react and look it up with the search_memory tool before "
    "concluding that nothing was recorded. "
)

AUTO_DURABLE_RULE = (
    "Independently of the action, list in durable_facts any stable "
    "facts from this conversation that future tasks should remember "
    "(client or project names, identifiers, budgets, deadlines, "
    "schedules, owners and reporting lines, decisions and their "
    "reasons, lasting preferences, corrections and changed facts), "
    "each as a self-contained statement. Emit one entry per fact: a "
    "turn that names a deadline, an owner and an amount yields three "
    "entries, never one combined sentence and never a partial "
    "selection — every fact stated must appear, because each entry "
    "is stored separately and anything you omit is forgotten. When "
    "unsure whether something is durable, include it: an extra "
    "stored fact is cheap, a forgotten one breaks a later task. A "
    "turn that names any project, person, number, date, place or "
    "decision is never 'nothing'. A correction is listed as the new "
    "statement, not skipped. Only pure small talk with no such "
    "content yields an empty array; the answer text itself is never "
    "a durable fact. "
)

AUTO_DURABLE_FACTS_SCHEMA_DESCRIPTION = (
    "Stable facts from this conversation that future "
    "tasks should remember, each a self-contained "
    "statement (client or project names, identifiers, "
    "budgets, deadlines, decisions, lasting "
    "preferences). Enumerate every distinct durable "
    "fact as its own entry: if the turn states a "
    "deadline, an owner and an amount, return three "
    "entries — never collapse several facts into one "
    "combined entry and never drop one, because each "
    "entry is stored separately and an omitted fact "
    "is lost forever. Use an empty array when nothing "
    "is durable, and never include transient chat, "
    "the final answer itself, or secrets, credentials "
    "and personal data that is not meant to persist."
)

STORE_MEMORY_DESCRIPTION = """Store a durable memory for future tasks.

Use this when you notice a UNIQUE, NON-OBVIOUS insight worth remembering, such as:
- A clear user preference or stable behavior pattern
- A non-obvious failure and how it was fixed
- A reusable strategy that is not routine
- A domain-specific insight that is hard to obtain otherwise
- A durable business or project fact a later task would need (a client or
  project name, an identifier, a budget, a deadline, a decision)

Call this once per fact. When one message states several durable facts (a deadline, an owner, an amount, a place), issue one call per fact instead of packing them into a single entry — each stored line is retrieved on its own, so a combined entry makes its parts unfindable, and an omitted fact is lost. When a message does carry durable facts, store all of them rather than a representative one.

If the material you are storing opens with a [to:<recipient>] delivery marker, copy that marker verbatim to the very start of the content parameter and keep the rest of the text intact. The marker is the delivery address: dropping or rewording it files the memory in your own store instead of the intended teammate's. Opening it with [share:dept] or [share:company] instead plants the note in the shared layer — your department's, or everyone's — rather than your own memory; both markers sit at the very start and are consumed by the service.

Do NOT store routine task completions, generic tool usage, common facts, obvious strategies, transient chatter, or secrets and credentials. Write the content as a self-contained statement understandable without this conversation.

If a similar memory already exists, the write is not dropped: it is merged
into that memory or attached under it, and the result says which happened.
Read the result before you speak: stored false or an error means nothing was
saved, so a 409 conflict is retried once as a new memory and, if that also
fails, you tell the user the fact was not saved instead of implying it was."""

SEARCH_MEMORY_DESCRIPTION = (
    "Search stored memories from previous tasks. Use this when earlier "
    "insights, user preferences, or known failure patterns could help the "
    "current step. Retrieval is semantic: write the query as one complete "
    "sentence stating what you are looking for — a person, a date, an "
    "amount, a place, a decision — because it matches meaning, not exact "
    "words, and the whole question retrieves better than a keyword list. "
    "Facts from one message are stored as separate lines, so search each "
    "fact you need on its own. If a query returns nothing, rephrase the "
    "same need in a different sentence once before concluding the fact "
    "was never stored. Each result sits in a tree of related memories: "
    "its parent (the cause or topic it hangs under), its children and "
    "its siblings are reachable — after a hit, follow up with another "
    "sentence built from that hit's own subject to climb, branch or "
    "move sideways along the tree until you have the context you need. "
    "Results include each memory's id, which update_memory and "
    "delete_memory take."
)

UPDATE_MEMORY_DESCRIPTION = (
    "Replace the content of a stored memory. Use this when a memory turns "
    "out to be outdated or contradicts what you observe now. Find the "
    "memory's id with search_memory first."
)

DELETE_MEMORY_DESCRIPTION = (
    "Permanently delete a stored memory. Use this when a memory is "
    "plainly wrong or harmful and cannot be fixed by update_memory, and "
    "whenever the user explicitly asks you to forget, drop or remove "
    "something you stored. Find the memory's id with search_memory first."
)

DELIVER_MEMORY_DESCRIPTION = (
    "Hand a note to a teammate's own memory instead of keeping it in "
    "yours: the recipient's agents retrieve it as if they had stored it "
    "themselves. Use it to pass a fact, decision or instruction to the "
    "agent or user who will act on it, addressed by username, external id "
    "or agent name. It lands in their inbox (read back with check_inbox), "
    "is rate-limited per recipient, and never exposes who else is copied. "
    "For your own durable facts use store_memory instead."
)

CHECK_INBOX_DESCRIPTION = (
    "Read the notes teammates have delivered to you — things other "
    "agents or users handed over for you to act on. Use this when the "
    "user asks what you were told, when a task refers to a handover, or "
    "before acting on an instruction that did not come from this "
    "conversation. Returns the newest deliveries first, each with its "
    "content, sender and time. Facts stored by your own past tasks are "
    "found with search_memory instead."
)
