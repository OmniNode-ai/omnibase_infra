# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for the evidence autoclose sweep (OMN-16106 first slice).

Wires: companion-merge -> dod_verify -> governed Done flip.

Pipeline
--------
1. Kill switch: if ``ONEX_AUTOCLOSE_DISABLED`` is set, do zero I/O and return.
2. Enumerate ``occ_repo`` PRs merged within ``lookback_hours`` via ``gh api``
   (paginated REST, newest-updated-first, bounded by ``max_companions``).
3. For each merged companion, extract its Evidence-Ticket binding from the
   changed-file list (``contracts/OMN-XXXXX.yaml``) and/or the PR title
   (``evidence(OMN-XXXXX)``). Zero or ambiguous (>1 distinct ticket id)
   bindings fail closed (skipped, never guessed).
4. For each uniquely-bound ticket: fetch its Linear state + labels. The
   ``close_if_done_label`` label or an already-completed/canceled state
   short-circuits to a skip (decision-only path stays manual).
5. Otherwise run the EXISTING verifier, ``onex skill dod_verify <ticket-id>``,
   dispatched from this process's own interpreter rather than re-resolved by
   ``uv run`` from a cwd (OMN-16846 — see ``_dod_verify_argv``, which carries
   the reason). Its stdout is parsed
   REGARDLESS of exit code — the CLI exits 1 on every genuine evidence gap
   while still printing a complete ``ModelSkillResult`` — and the verdict is
   read from whichever of the CLI's two declared result arms the receipt's
   own ``result_model`` names: flat on ``result`` for a success-like run,
   nested at ``result.terminal_payload`` otherwise (OMN-16961 —
   see ``_extract_dod_verify_verdict``). FLIP requires all of: the verifier's own
   ``status == "verified"``, ``total_checks > 0``, zero failed, at least one
   verified check, and ``verified_count + non_probative_count ==
   total_checks`` (OMN-16821 — see step 9). Anything else that still reached
   a verdict is a GAP. Output with no parseable verdict in it fails closed as
   an error (recorded, never flips).
6. AC-COVERAGE GUARD (OMN-16736). A green dod_verify is necessary, not
   sufficient. dod_verify verifies the checks declared in the ticket's OCC
   contract; an acceptance criterion written only into the Linear
   description is structurally invisible to it, so "3/3 verified, 0 failed"
   says nothing at all about a fourth criterion nobody ever encoded — the
   OMN-14362 lesson. So before ANY flip path (dry-run included) the
   ticket's description is re-read: an unchecked markdown checkbox, or an
   acceptance-criteria section listing more items than dod_verify had
   checks, withholds the flip and records GAP_AC_COVERAGE naming the
   criteria. Conservative in exactly one direction on purpose — a false
   hold costs a comment and a human glance, a false flip writes an unearned
   Done onto the board.
7. PROOF-CLASS GUARD (OMN-15911). A green tally does not say what the green
   legs PROVED. Before OMN-15911 a `gh pr view --json state` read and an
   executed test suite both terminated in the same `verified`, so
   ``verified_count == total_checks`` was satisfiable entirely by "the PR
   merged". dod_verify now classifies every check verdict and reports
   ``behavior_proving_count`` (VERIFIED and BEHAVIOR); a flip requires it to
   be >= 1, and a zero on an otherwise-clean run records
   GAP_NO_BEHAVIOR_PROOF. A verdict carrying no such field at all (an
   omnimarket predating the change — i.e. every historical receipt) is an
   ERROR, not an inference.
8. COMMENT IDEMPOTENCY (OMN-16808). Enumeration is a bare
   ``now - lookback_hours`` window with no cursor or watermark, and
   ``seen_tickets`` is a per-run local, so one merged companion sits inside
   several consecutive scheduled windows (``*/30`` cron against a multi-hour
   lookback). Before ANY gap comment the ticket's existing comments are read
   and matched against a stable marker embedded in every comment the sweep
   writes; an equivalent statement already on the ticket is recorded as
   SKIPPED_DUPLICATE_COMMENT and not repeated. Fail-closed: a comment history
   that cannot be read is ERROR_LINEAR_API and writes nothing, because "I
   could not check" must never resolve to "so I will post". The FLIPPED path
   needs no such guard — it terminates in a completed state that
   SKIPPED_ALREADY_DONE short-circuits on the next run.
9. VERDICT-BEARING DENOMINATOR (OMN-16821). The flip equality used to be
   ``verified_count == total_checks``, and it was unsatisfiable for most of
   the corpus. ``total_checks`` counts every check that carries a verdict,
   and ``non_probative`` (OMN-15391 — executed, exited 0, and its exit status
   cannot depend on the product change) IS one of those verdicts. So a
   ``non_probative`` entry counted in the denominator and never in the
   numerator, and one of them was enough to hold a ticket forever no matter
   how strong the rest of the evidence was. ``gh pr view --json state``
   surrogates are the most common check shape in the autobound OCC corpus,
   so this was the reason the flip path had never fired once: OMN-16260
   measured ``verified`` / 12 total / 6 verified / 0 failed / 1
   behavior-proving and still could not flip, purely on the arithmetic —
   while being classified GAP_POSTED, i.e. told under ``--apply`` that "not
   all ACs are receipt-proven" by a verifier that had just said the opposite.
   The honest equality is ``verified_count + non_probative_count ==
   total_checks``, which is the OMN-15390 precedent applied one axis over
   (``total_checks`` already excludes ``superseded`` upstream, on exactly the
   reasoning that an entry carrying no product-dependent verdict does not
   belong in a verdict-bearing denominator). Strictness is unchanged in every
   other direction and each is pinned by a test: a real ``failed`` still
   gaps; a ``skipped`` check still gaps (it is neither verified nor
   non-probative, so it breaks the equality); an ALL-non-probative contract
   still gaps twice over — dod_verify's own status is ``skipped``, and
   ``verified_count > 0`` is required so the equality can never be satisfied
   by a denominator made entirely of provenance. The OMN-15911 behavior
   conjunct and the OMN-16736 AC-coverage guard both run AFTER this and are
   untouched by it. A verdict omitting ``non_probative_count`` entirely
   contributes 0 and collapses back to the old, stricter rule — a fallback
   that can only withhold a flip, never grant one.
10. ``apply=False`` (the default) performs every read above but never calls
   a Linear mutation — every decision is logged as "would-do". ``apply=True``
   performs the real ``issueUpdate``/``commentCreate`` mutation. The dedup
   read runs in BOTH modes: it is a read, so DRY-RUN stays zero-write, and a
   DRY-RUN report is then an honest preview of what ``--apply`` would post.

Non-blocking Design
--------------------
Per-ticket failures (Linear API errors, dod_verify crashes) are recorded in
the outcome list and do not abort the sweep — only a sweep-level failure
(GitHub enumeration itself failing) sets ``result.success = False``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_outcome import (
    ModelEvidenceAutocloseOutcome,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_result import (
    ModelEvidenceAutocloseSweepResult,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

__all__ = ["HandlerEvidenceAutocloseSweep"]


def _dod_verify_argv(ticket_id: str) -> list[str]:
    """Argv that dispatches the verifier from THIS process's own environment.

    OMN-16846. This used to be ``["uv", "run", "onex", "skill", "dod_verify",
    ticket_id]``, which does not name an environment at all — ``uv run``
    re-resolves the project at the subprocess's cwd and uses that project's
    venv. The sweep's cwd is the omnibase_infra product clone, so the verifier
    landed in the PRODUCT venv, and the only way to make the verifier
    resolvable was to co-install its provider (omnimarket, which ships
    ``node_dod_verify``) into that same venv.

    That is the collision. dod_verify's behaviour checks are ``uv run pytest``
    with ``cwd: "${OMNI_HOME}/omnibase_infra"``, so they resolve the very same
    venv, where ``tests/conftest.py`` calls ``assert_venv_purity()``
    (OMN-15620) and correctly refuses an undeclared ``onex.nodes`` provider.
    Both halves are right; only their collapse onto one venv is wrong. Run
    33194402437 is the receipt: all three OMN-16759 behaviour checks recorded
    ``FAILED ... Canonical venv is IMPURE``, ``behavior_proving=0``, without
    a single test having executed.

    Dispatching through ``sys.executable``'s sibling ``onex`` makes the
    verifier's environment a property of how the SWEEP was composed rather
    than of where it happens to be standing. The dispatch venv can then carry
    the co-installed omnimarket while the product clone's venv stays pure and
    the behaviour checks actually run. It is also strictly more determinate
    for the existing local path, where both resolved to the same venv anyway.

    A missing sibling ``onex`` raises ``FileNotFoundError`` at exec time,
    which the caller already converts into a named per-ticket error; the
    sweep job additionally asserts the binary resolves before any ticket is
    scanned, so the loud failure comes first.
    """
    return [str(Path(sys.executable).parent / "onex"), "skill", "dod_verify", ticket_id]


# Injectable subprocess-runner signatures (real impls call `gh`/the dispatch
# venv's `onex`; tests inject fakes with the same shape).
TypeRunGhCommand = Callable[[list[str], float], Awaitable[tuple[object | None, str]]]
TypeRunDodVerifyCommand = Callable[
    [str, str, float], Awaitable[tuple[dict[str, object] | None, int, str]]
]

# Kill switch env var (checked first, unconditionally — OMN-16106).
_KILL_SWITCH_ENV_VAR = "ONEX_AUTOCLOSE_DISABLED"

# The key GitHub's `GET /repos/{owner}/{repo}/pulls/{number}/files` response
# uses for a changed file's repo-relative path. It is "filename" — NOT "path",
# which belongs to the Contents/trees APIs and never appears here (OMN-16736).
_GH_PR_FILE_PATH_KEY = "filename"

# Where `onex skill dod_verify <ticket>` actually puts its verdict.
#
# `receipt_mode._run_and_emit` prints ONE of TWO receipt arms, and which one
# is decided by the run's own outcome (OMN-16961):
#
#   success-like AND a handler result exists
#       -> `result` IS the handler's own model, FLAT. The verdict keys
#          (status, total_checks, verified_count, ...) sit directly on
#          `result`, and there is no `terminal_payload` key anywhere.
#          `result_model` = ModelDodVerifyState.
#   anything else
#       -> `result` is a ModelReceiptRuntimeSummary (workflow_result,
#          exit_code, workflow, terminal_payload, handler_result, error,
#          capture_log) and the verdict is nested at `result.terminal_payload`.
#          `result_model` = ModelReceiptRuntimeSummary.
#
# OMN-16736 read the second arm only. Its constant was verified against a
# single live capture (tests/fixtures/omn16736/) that happened to be a
# `status: failed` run — the summary arm. Generalising it to every outcome
# inverted the sweep: dod_verify's workflow result is success-like exactly
# when its verdict is `verified`, so the arm the reader could not parse was
# precisely the FLIP-ELIGIBLE one. Run 33258391128 recorded 10 of 19
# verdict-eligible tickets as `error_verify_unparseable` at `exit_code=0`
# while the diagnose step — which falls back to the envelope itself — read
# full verdicts from the same bytes. `tickets_flipped` could not leave 0.
#
# `result_model` is the receipt's own declared type tag, so reading it is
# using the contract rather than sniffing the shape. Both arms are captured
# verbatim at tests/fixtures/omn16961/ (OMN-16961).
_RECEIPT_SUMMARY_RESULT_MODEL = (
    "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
)
_DOD_VERIFY_STATE_RESULT_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)
_DOD_VERIFY_VERDICT_KEY = "terminal_payload"
# dod_verify's own terminal status for "every verdict-bearing check passed".
# `EnumDodVerifyStatus.VERIFIED` in omnimarket.
_DOD_VERIFY_STATUS_VERIFIED = "verified"
# How many passing checks executed the claimed behavior, as opposed to reading
# PR/merge state or standing in as a surrogate. `ModelDodVerifyState
# .behavior_proving_count` in omnimarket, added by OMN-15911. Absent on any
# verdict produced by an omnimarket predating that change — which is every
# receipt in the existing corpus, so its absence is an ERROR (nothing to
# decide on), never an inference in either direction.
_DOD_VERIFY_BEHAVIOR_KEY = "behavior_proving_count"
# How many checks executed, exited 0, and could not have exited otherwise for
# a product reason — a bare `gh pr view` (green for every PR on GitHub) or a
# ticket-independent foreign suite. `ModelDodVerifyState.non_probative_count`
# in omnimarket, added by OMN-15391. It is a VERDICT, so it counts in
# `total_checks`; it is not a PROOF, so it never counts in `verified_count`.
# Both facts are correct, and reconciling them is what OMN-16821 is about (see
# step 9 of the module docstring). Absent on a verifier predating OMN-15391,
# where it contributes 0 and the predicate degrades to the older, stricter
# equality — a fallback that can only withhold a flip, never grant one.
_DOD_VERIFY_NON_PROBATIVE_KEY = "non_probative_count"

# Evidence-Ticket binding patterns.
_CONTRACT_FILE_RE = re.compile(r"^contracts/(OMN-\d+)\.yaml$")
_TITLE_EVIDENCE_RE = re.compile(r"evidence\((OMN-\d+)\)", re.IGNORECASE)

_LINEAR_API_URL = "https://api.linear.app/graphql"  # url-authority-ok: fixed public GraphQL API, no ONEX routing authority

# `description` is fetched for the AC-coverage guard below (OMN-16736): the
# acceptance criteria dod_verify CANNOT see are exactly the ones that live only
# here, in the ticket body, and never made it into the OCC contract.
_ISSUE_QUERY = """
query GetIssue($id: String!) {
  issue(id: $id) {
    id
    identifier
    description
    state { id name type }
    labels(first: 50) { nodes { name } }
    team { id }
  }
}
"""

_TEAM_STATES_QUERY = """
query TeamStates($teamId: String!) {
  team(id: $teamId) {
    id
    states(first: 50) { nodes { id name type } }
  }
}
"""

_ISSUE_UPDATE_STATE_MUTATION = """
mutation UpdateIssueState($issueId: String!, $stateId: String!) {
  issueUpdate(id: $issueId, input: { stateId: $stateId }) {
    success
  }
}
"""

_COMMENT_CREATE_MUTATION = """
mutation CreateComment($issueId: String!, $body: String!) {
  commentCreate(input: { issueId: $issueId, body: $body }) {
    success
  }
}
"""

# OMN-16808. The read half of read-before-write: what has this sweep already
# said on this ticket? Paginated, because a partial history is indistinguishable
# from "no marker present" and would resolve straight into a duplicate post.
_ISSUE_COMMENTS_QUERY = """
query IssueComments($id: String!, $after: String) {
  issue(id: $id) {
    id
    comments(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { id body }
    }
  }
}
"""

# Page cap for the comment-history read. Exhausting it WITHOUT reaching the end
# of the connection is an unreadable history, not an empty one — the caller
# fails closed. 5 x 100 comfortably exceeds any real ticket's comment count;
# a ticket that somehow exceeds it stops receiving sweep comments rather than
# receiving duplicates, which is the correct direction to be wrong in.
_MAX_COMMENT_PAGES = 5


async def _reap_timed_out_process(proc: asyncio.subprocess.Process) -> None:
    """Kill and reap a subprocess whose ``communicate()`` await timed out.

    ``asyncio.wait_for`` cancels the *await*, not the child process: without
    this, a `gh`/`onex` invocation that hangs past its timeout keeps running
    with its stdout/stderr pipes held open, leaking a process per timeout on
    every scheduled sweep tick.
    """
    if proc.returncode is not None:
        return
    proc.kill()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        logger.warning(
            "Timed-out subprocess (pid=%s) did not exit after kill()", proc.pid
        )


def _as_int(value: object) -> int:
    """Best-effort int coercion for a loosely-typed dod_verify JSON field."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _extract_dod_verify_verdict(
    receipt: dict[str, object],
) -> tuple[dict[str, object] | None, str]:
    """Return the dod_verify verdict from either declared receipt arm.

    OMN-16961. `onex skill dod_verify` prints two structurally different
    receipts (see `_RECEIPT_SUMMARY_RESULT_MODEL` above for the branch that
    picks between them). Both are legitimate, both are declared, and the
    receipt names which one it is in `result_model`. This reader dispatches
    on that tag and on nothing else.

    It does NOT sniff for whichever key happens to be present. A receipt
    whose `result_model` is absent or unrecognised is REFUSED by name — the
    sweep would rather stop than guess at an undeclared shape, because a
    wrong guess here is a Done flip on evidence nobody read (OMN-15715 /
    OMN-16832 AC2). Same for a declared arm that carries no `total_checks`:
    that is a dispatch which produced output but reached no verdict, and it
    stays an error, never a 0/0 "gap" that reads like a ticket problem.

    Returns ``(verdict, "")`` on success and ``(None, reason)`` on refusal,
    where ``reason`` names the specific cause rather than the generic "no
    verdict was reached".
    """
    result_model = receipt.get("result_model")
    result = receipt.get("result")
    result = result if isinstance(result, dict) else {}

    if not isinstance(result_model, str) or not result_model.strip():
        return None, (
            "the receipt declares no `result_model`, so which of the two "
            "`onex skill` result arms it carries cannot be established — "
            "refusing to guess at an undeclared shape."
        )

    if result_model == _RECEIPT_SUMMARY_RESULT_MODEL:
        verdict = result.get(_DOD_VERIFY_VERDICT_KEY)
        if not isinstance(verdict, dict):
            return None, (
                f"the receipt is a `{_RECEIPT_SUMMARY_RESULT_MODEL}` whose "
                f"`result.{_DOD_VERIFY_VERDICT_KEY}` is absent — the runtime "
                "emitted no terminal event, so the verifier genuinely reached "
                "no verdict."
            )
        arm = f"result.{_DOD_VERIFY_VERDICT_KEY}"
    elif result_model == _DOD_VERIFY_STATE_RESULT_MODEL:
        # Success arm: `result` IS the verdict, flat. No nesting exists.
        verdict = result
        arm = "result"
    else:
        return None, (
            f"unrecognised dod_verify receipt `result_model` {result_model!r} "
            f"— expected {_DOD_VERIFY_STATE_RESULT_MODEL!r} (success arm) or "
            f"{_RECEIPT_SUMMARY_RESULT_MODEL!r} (runtime-summary arm). The "
            "receipt contract drifted; refusing to infer a verdict from an "
            "unknown shape."
        )

    if "total_checks" not in verdict:
        return None, (
            f"the receipt declares `{result_model}` but its `{arm}` carries no "
            "`total_checks` — output was produced, no verdict was reached."
        )
    return verdict, ""


def _extract_ticket_binding(title: str, files: list[str]) -> tuple[str | None, bool]:
    """Extract the Evidence-Ticket binding for one merged companion.

    Returns:
        ``(ticket_id, ambiguous)``. ``ticket_id`` is ``None`` when no
        binding was found at all (missing binding, fail closed).
        ``ambiguous`` is True when more than one distinct ticket id was
        found across the contract-file and title signals (fail closed —
        never guess which one is authoritative).
    """
    found: set[str] = set()
    for path in files:
        match = _CONTRACT_FILE_RE.match(path)
        if match:
            found.add(match.group(1))
    title_match = _TITLE_EVIDENCE_RE.search(title)
    if title_match:
        found.add(title_match.group(1).upper())

    if not found:
        return None, False
    if len(found) > 1:
        return None, True
    return next(iter(found)), False


# -- AC-coverage guard (OMN-16736; the OMN-14362 lesson) -------------------
#
# dod_verify verifies the checks declared in the ticket's OCC contract. An
# acceptance criterion that was written into the Linear description and never
# transcribed into that contract is STRUCTURALLY INVISIBLE to it: dod_verify
# reports 3/3 verified, 0 failed, and says nothing whatsoever about the fourth
# criterion nobody encoded. Flipping Done on that reading is a wrong flip, not
# a conservative one.
#
# So the flip path is gated on a second, cheap, purely textual read of the
# description. It is deliberately CONSERVATIVE IN ONE DIRECTION: a false hold
# costs a comment and a human glance; a false flip writes an unearned Done onto
# the board, which is the exact failure the whole OMN-16106 mechanism exists to
# avoid producing at scale.

# An unchecked GitHub-flavoured-markdown task item, anywhere in the body.
# Deliberately not scoped to an acceptance-criteria section: an unchecked box
# is an author's own "not done yet" marker wherever it appears.
_UNCHECKED_TASK_RE = re.compile(r"^[ \t]*[-*+][ \t]+\[[ \t]\][ \t]*(.*)$", re.MULTILINE)

# A bullet or numbered list item.
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+(.*)$")
# An `AC1: ...` / `AC-2 ...` line with no bullet at all -- a common shape in
# these tickets that a list-item-only parser would silently count as zero.
_AC_ITEM_RE = re.compile(r"^[ \t]*(AC[-_ ]?\d+\b.*)$", re.IGNORECASE)
# A leading `[ ]` / `[x]` task marker, stripped from item text for readability.
_TASK_MARKER_RE = re.compile(r"^\[[ \t xX]\][ \t]*")
# Leading enumeration on a heading line: `3. Acceptance criteria`.
_HEADING_ENUM_RE = re.compile(r"^\d+[.)]\s*")

# Heading texts that open an acceptance-criteria section, after normalization.
_AC_HEADING_TEXTS = frozenset(
    {
        "acceptance criteria",
        "acceptance criteria (ac)",
        "acceptance criterion",
        "acceptance",
        "ac",
        "acs",
    }
)

# Cap on how many uncovered criteria are spelled out in the Linear comment.
# A description with 40 unchecked boxes does not need 40 quoted lines to make
# the point, and an unbounded splice is how a comment body hits an API limit.
_MAX_UNCOVERED_LISTED = 20


def _is_markdown_heading(line: str) -> bool:
    return line.lstrip().startswith("#")


def _is_ac_heading(line: str) -> bool:
    """True when ``line`` reads as an 'Acceptance criteria' heading.

    Tolerates ``## Acceptance Criteria``, ``**Acceptance criteria:**``,
    ``### 3. Acceptance criteria`` and bare ``AC``.
    """
    text = line.strip()
    if not text:
        return False
    text = text.lstrip("#").strip()
    text = text.strip("*_").strip()
    text = _HEADING_ENUM_RE.sub("", text)
    text = text.rstrip(":").strip()
    text = text.strip("*_").strip()
    return text.casefold() in _AC_HEADING_TEXTS


def _acceptance_criteria_items(description: str) -> list[str]:
    """Items listed under an acceptance-criteria heading in ``description``.

    The section runs from the heading to the next markdown heading (or the end
    of the body). A non-``#`` heading -- a bold pseudo-heading, say -- does not
    close the section, so the count can be an OVER-count. That direction is
    deliberate: over-counting holds a flip, under-counting releases one.
    """
    items: list[str] = []
    in_section = False
    for line in description.splitlines():
        if _is_ac_heading(line):
            in_section = True
            continue
        if not in_section:
            continue
        if _is_markdown_heading(line):
            break
        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            text = _TASK_MARKER_RE.sub("", list_match.group(1)).strip()
            if text:
                items.append(text)
            continue
        ac_match = _AC_ITEM_RE.match(line)
        if ac_match:
            text = ac_match.group(1).strip()
            if text:
                items.append(text)
    return items


def _ac_coverage_gap(
    description: str, total_checks: int
) -> tuple[str, tuple[str, ...]]:
    """Decide whether ``description`` carries criteria dod_verify did not cover.

    Returns ``(reason, uncovered)``. An empty ``reason`` means no gap was
    found and the flip may proceed. Two rules, checked in order:

    1. ANY unchecked markdown task item (``- [ ]``) -- the author's own
       "not done" marker, which a contract verifier never reads.
    2. The acceptance-criteria section lists MORE items than dod_verify had
       checks. Which specific ones are uncovered cannot be known from a count,
       so every listed item is named and the arithmetic is stated.

    An empty/absent description is NOT a gap: Linear returns null for a ticket
    with no body, and treating "no criteria written down" as "criteria we
    cannot read" would turn the guard into a blanket hold on every such ticket.
    """
    if not description.strip():
        return "", ()

    unchecked = tuple(
        text
        for text in (
            match.group(1).strip() for match in _UNCHECKED_TASK_RE.finditer(description)
        )
        if text
    )
    if unchecked:
        return (
            f"{len(unchecked)} unchecked checkbox item(s) in the Linear "
            "description. dod_verify only verifies the checks declared in the "
            "OCC contract; a criterion that lives only in the ticket body is "
            "invisible to it, so a 0-failed run says nothing about these.",
            unchecked,
        )

    items = tuple(_acceptance_criteria_items(description))
    if len(items) > total_checks:
        return (
            f"The Linear description's acceptance-criteria section lists "
            f"{len(items)} item(s) but dod_verify covered only {total_checks} "
            f"check(s) -- at least {len(items) - total_checks} criterion(s) is "
            "not receipt-proven. Which ones cannot be told from a count, so all "
            "listed items are named below.",
            items,
        )

    return "", ()


# -- comment idempotency marker (OMN-16808) --------------------------------
#
# Every comment the sweep writes ends with an HTML-comment marker. Linear
# renders markdown, so the marker is invisible in the UI and exact-matchable in
# the raw body — the sweep's own durable record that it has already made this
# statement on this ticket, carried by the projection (the ticket) rather than
# by the job. That is the same shape as every other piece of truth here: state
# is read back from the surface it was written to, not held in the runner.
#
# The key is (gap class, verdict fingerprint) — NOT the companion PR. Two
# facts drive that choice:
#   * the same verdict re-derived from a later companion is the SAME statement,
#     and keying on the companion would let every new OCC merge re-open the
#     same gap comment;
#   * a CHANGED verdict (3/6 -> 5/6, or GAP_POSTED -> GAP_NO_BEHAVIOR_PROOF) is
#     new information, gets a different fingerprint, and is posted.
# So each (ticket, gap class) carries at most one live statement, refreshed
# when the underlying verdict moves and silent when it does not.
_SWEEP_COMMENT_MARKER_PREFIX = "onex-autoclose-sweep"
_SWEEP_COMMENT_MARKER_VERSION = "v1"


def _gap_shortfall(
    *,
    verify_status: str,
    total_checks: int,
    verified_count: int,
    failed_count: int,
    non_probative_count: int,
) -> str:
    """Name the shortfall this gap ACTUALLY found (OMN-16821 AC4).

    Every gap used to be reported as "not all ACs are receipt-proven". For a
    run that FAILED a check that is true. For a run dod_verify itself called
    ``verified`` with zero failures it is a false statement about the ticket,
    written into Linear by the mechanism — the same class of wrong statement
    OMN-16736 (AC coverage) and OMN-15911 (proof class) were each opened to
    stop. The branches below are ordered most-specific-first and each states
    the fact that actually withheld the flip.
    """
    if failed_count > 0:
        return "not all ACs are receipt-proven."
    if verified_count == 0 and non_probative_count > 0:
        # OMN-15391's refusal. Nothing went wrong; nothing was proven either.
        return (
            f"no check carried a probative verdict — {non_probative_count} "
            f"non-probative of {total_checks}, and dod_verify's own terminal "
            f"status is {verify_status!r}."
        )
    if verify_status != _DOD_VERIFY_STATUS_VERIFIED:
        return (
            f"dod_verify's own terminal status is {verify_status!r}, not "
            f"{_DOD_VERIFY_STATUS_VERIFIED!r}."
        )
    if total_checks == 0:
        return "the contract declared no verdict-bearing check at all."
    unaccounted = total_checks - verified_count - non_probative_count
    if unaccounted > 0:
        return (
            f"{unaccounted} check(s) reached no verdict (neither verified nor "
            "non-probative) — a check that never ran proves nothing."
        )
    # Unreachable while `all_verified` is the exact negation of this function's
    # inputs; kept as an honest catch-all rather than an assertion, because a
    # future conjunct added to the predicate and not to this branch table would
    # otherwise silently reuse whichever clause happened to be last.
    return "not all ACs are receipt-proven."


def _sweep_comment_marker(
    decision: EnumEvidenceAutocloseDecision, fingerprint_parts: tuple[str, ...]
) -> str:
    """Build the stable per-(ticket, gap class, verdict) comment marker."""
    digest = hashlib.sha256("\x1f".join(fingerprint_parts).encode("utf-8")).hexdigest()[
        :16
    ]
    return (
        f"<!-- {_SWEEP_COMMENT_MARKER_PREFIX} {_SWEEP_COMMENT_MARKER_VERSION} "
        f"class={decision.value} fingerprint={digest} -->"
    )


def _format_uncovered(uncovered: tuple[str, ...]) -> str:
    """Render the uncovered criteria as a bounded markdown list."""
    shown = uncovered[:_MAX_UNCOVERED_LISTED]
    lines = [f"- {text}" for text in shown]
    remaining = len(uncovered) - len(shown)
    if remaining > 0:
        lines.append(f"- ... and {remaining} more (truncated)")
    return "\n".join(lines)


class _LinearClient:
    """Minimal Linear GraphQL client scoped to this sweep's needs.

    Reads ``LINEAR_API_KEY`` from the environment when not passed
    explicitly (mirrors ``GitHubTransport.__init__``'s ``GH_PAT`` fallback
    in ``omnibase_infra.adapters.github.adapter_github_client``).
    """

    # OMN-14951 gap 2: self-declared secret-ish env-var names read by this
    # boundary file (see scripts/check-env-reads.sh's check_secret_name_declarations).
    required_secrets: tuple[str, ...] = ("LINEAR_API_KEY",)

    def __init__(self, api_key: str | None = None, timeout: float = 15.0) -> None:
        self._api_key = (
            api_key if api_key is not None else os.environ.get("LINEAR_API_KEY", "")
        )
        self._timeout = timeout

    async def _query(
        self, query: str, variables: dict[str, object]
    ) -> dict[str, object] | None:
        if not self._api_key:
            logger.warning("LINEAR_API_KEY is not set — cannot call Linear API.")
            return None
        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {"query": query, "variables": variables}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    _LINEAR_API_URL, json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Linear API call failed: %s", sanitize_error_message(exc))
            return None
        if data.get("errors"):
            logger.warning("Linear API returned GraphQL errors: %s", data["errors"])
            return None
        result = data.get("data")
        return result if isinstance(result, dict) else None

    async def fetch_issue(self, ticket_id: str) -> dict[str, object] | None:
        """Fetch id/state/labels/team for a ticket. None on any failure."""
        data = await self._query(_ISSUE_QUERY, {"id": ticket_id})
        if data is None:
            return None
        issue = data.get("issue")
        return issue if isinstance(issue, dict) else None

    async def fetch_done_state_id(self, team_id: str) -> str | None:
        """Resolve the team's completed-type state named 'Done'. None if absent."""
        data = await self._query(_TEAM_STATES_QUERY, {"teamId": team_id})
        if data is None:
            return None
        team = data.get("team")
        states_conn = team.get("states") if isinstance(team, dict) else None
        nodes = states_conn.get("nodes") if isinstance(states_conn, dict) else None
        if not isinstance(nodes, list):
            return None
        for state in nodes:
            if not isinstance(state, dict):
                continue
            if (
                state.get("type") == "completed"
                and str(state.get("name", "")).strip().lower() == "done"
            ):
                state_id = state.get("id")
                return str(state_id) if state_id else None
        return None

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        """Flip an issue to the given workflow state. False on any failure."""
        data = await self._query(
            _ISSUE_UPDATE_STATE_MUTATION, {"issueId": issue_id, "stateId": state_id}
        )
        if data is None:
            return False
        update = data.get("issueUpdate")
        return bool(isinstance(update, dict) and update.get("success"))

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        """Every existing comment body on an issue, or None if unreadable.

        None means "could not determine", NEVER "there are none" (OMN-16808).
        The caller uses this to decide whether it has already said something on
        this ticket, so an empty tuple and a failed read must not collapse into
        the same value — one releases a write, the other must block it.

        Fails closed on: an API/GraphQL failure, a payload shape this code
        cannot interpret, a cursor the connection did not supply, and a history
        longer than ``_MAX_COMMENT_PAGES`` pages (an unread tail could hold the
        marker).
        """
        bodies: list[str] = []
        cursor: str | None = None
        for _ in range(_MAX_COMMENT_PAGES):
            data = await self._query(
                _ISSUE_COMMENTS_QUERY, {"id": issue_id, "after": cursor}
            )
            if data is None:
                return None
            issue = data.get("issue")
            if not isinstance(issue, dict):
                return None
            connection = issue.get("comments")
            if not isinstance(connection, dict):
                return None
            nodes = connection.get("nodes")
            if not isinstance(nodes, list):
                return None
            for node in nodes:
                if isinstance(node, dict):
                    body = node.get("body")
                    if isinstance(body, str):
                        bodies.append(body)
            page_info = connection.get("pageInfo")
            page_info = page_info if isinstance(page_info, dict) else {}
            if not page_info.get("hasNextPage"):
                return tuple(bodies)
            end_cursor = page_info.get("endCursor")
            if not isinstance(end_cursor, str) or not end_cursor:
                return None
            cursor = end_cursor
        return None

    async def create_comment(self, issue_id: str, body: str) -> bool:
        """Post a comment on an issue. False on any failure."""
        data = await self._query(
            _COMMENT_CREATE_MUTATION, {"issueId": issue_id, "body": body}
        )
        if data is None:
            return False
        created = data.get("commentCreate")
        return bool(isinstance(created, dict) and created.get("success"))


class HandlerEvidenceAutocloseSweep:
    """Sweep merged OCC companions and flip/comment on their bound tickets."""

    def __init__(
        self,
        linear_client: _LinearClient | None = None,
        autoclose_disabled: bool | None = None,
        run_gh_command: TypeRunGhCommand | None = None,
        run_dod_verify_command: TypeRunDodVerifyCommand | None = None,
    ) -> None:
        # ``autoclose_disabled`` mirrors GitHubTransport's env-fallback-in-
        # __init__ precedent: read at construction time, override injectable
        # for tests. Re-checked defensively at the top of handle() too so a
        # zero-arg contract-driven construction can never silently skip it.
        #
        # `gh api` timeout is NOT stored here (unlike a prior revision) --
        # it is contract-exposed as request.gh_timeout_seconds and threaded
        # through at each call site, the same pattern already used for
        # request.dod_verify_timeout_seconds below. See OMN-16106.
        self._linear = linear_client if linear_client is not None else _LinearClient()
        self._autoclose_disabled_ctor = (
            autoclose_disabled
            if autoclose_disabled is not None
            else bool(os.environ.get(_KILL_SWITCH_ENV_VAR, ""))
        )
        # Injectable subprocess runners for tests (fake gh / dod_verify).
        self._run_gh_command = run_gh_command or self._run_gh_command_real
        self._run_dod_verify_command = (
            run_dod_verify_command or self._run_dod_verify_command_real
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    # -- subprocess runners (real) -------------------------------------

    async def _run_gh_command_real(
        self, args: list[str], timeout: float
    ) -> tuple[object | None, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            # Process creation itself can raise (missing executable, invalid
            # cwd) before there is any `proc` to reap — must be caught here,
            # not only around `communicate()` below.
            return None, f"OS error launching {' '.join(args)}: {exc}"
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            await _reap_timed_out_process(proc)
            return None, f"Timeout running: {' '.join(args)}"
        except OSError as exc:
            await _reap_timed_out_process(proc)
            return None, f"OS error running {' '.join(args)}: {exc}"
        if proc.returncode != 0:
            return None, stderr.decode(errors="replace").strip()
        try:
            return json.loads(stdout.decode()), ""
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON from {' '.join(args)}: {exc}"

    async def _run_dod_verify_command_real(
        self, ticket_id: str, cwd: str, timeout: float
    ) -> tuple[dict[str, object] | None, int, str]:
        # OMN-16846: resolved from THIS process's interpreter, not re-resolved
        # by `uv run` from the cwd. See `_dod_verify_argv`.
        args = _dod_verify_argv(ticket_id)
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=cwd or None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            # Process creation itself can raise (missing executable, invalid
            # cwd) before there is any `proc` to reap — must be caught here,
            # not only around `communicate()` below.
            return None, -1, f"OS error launching dod_verify for {ticket_id}: {exc}"
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            await _reap_timed_out_process(proc)
            return None, -1, f"Timeout running dod_verify for {ticket_id}"
        except OSError as exc:
            await _reap_timed_out_process(proc)
            return None, -1, f"OS error running dod_verify for {ticket_id}: {exc}"
        exit_code = proc.returncode if proc.returncode is not None else -1
        stderr_text = stderr.decode(errors="replace").strip()
        # Parse stdout REGARDLESS of exit code (OMN-16736). `onex skill` exits
        # non-zero whenever the verdict is `failed` — i.e. on every genuine
        # evidence GAP — while still printing a complete, valid ModelSkillResult
        # on stdout. A prior revision discarded stdout the moment the exit code
        # was non-zero, so every real gap was misrecorded as
        # ERROR_VERIFY_NONZERO_EXIT ("the verifier crashed") instead of
        # GAP_POSTED ("the ticket is not proven"). Those are different facts and
        # only one of them is actionable by the ticket's owner.
        try:
            parsed = json.loads(stdout.decode(errors="replace"))
        except json.JSONDecodeError as exc:
            # No parseable verdict. Prefer stderr when the process also failed —
            # that is where a dispatch error (missing node, bad contract) lands.
            return (
                None,
                exit_code,
                stderr_text if exit_code != 0 else f"Invalid dod_verify JSON: {exc}",
            )
        if not isinstance(parsed, dict):
            return None, exit_code, "dod_verify output was not a JSON object"
        return parsed, exit_code, ""

    # -- GitHub enumeration ----------------------------------------------

    async def _fetch_merged_companions(
        self, repo: str, since_iso: str, max_companions: int, gh_timeout_seconds: int
    ) -> tuple[list[dict[str, object]], str]:
        """Paginated `gh api` enumeration of merged PRs, newest-updated-first."""
        merged: list[dict[str, object]] = []
        per_page = 100
        max_pages = 20  # safety cap against unbounded pagination
        for page in range(1, max_pages + 1):
            path = (
                f"repos/{repo}/pulls?state=closed&sort=updated"
                f"&direction=desc&per_page={per_page}&page={page}"
            )
            batch, error = await self._run_gh_command(
                ["gh", "api", path], gh_timeout_seconds
            )
            if batch is None:
                return merged, error
            if not isinstance(batch, list) or not batch:
                break
            stop_paginating = False
            for pr in batch:
                if not isinstance(pr, dict):
                    continue
                updated_at = str(pr.get("updated_at") or "")
                if updated_at and updated_at < since_iso:
                    stop_paginating = True
                    continue
                merged_at = pr.get("merged_at")
                if merged_at and str(merged_at) >= since_iso:
                    merged.append(pr)
                    if len(merged) >= max_companions:
                        return merged[:max_companions], ""
            if stop_paginating or len(batch) < per_page:
                break
        return merged, ""

    async def _fetch_pr_files(
        self, repo: str, number: int, gh_timeout_seconds: int
    ) -> tuple[list[str], str]:
        """Fetch a PR's changed-file paths. Never silently degrades to empty.

        Returns ``(files, error)``. On a genuine fetch failure ``error`` is
        non-empty and ``files`` is empty — callers MUST treat that as
        ERROR_GITHUB_API, never as "this companion touched zero files" (which
        would fall through to a title-only binding match a real file listing
        might have disambiguated or contradicted).

        OMN-16736: this read used ``item["path"]``. ``GET /repos/{owner}/{repo}
        /pulls/{number}/files`` keys every entry on ``filename``; ``path`` is
        the Contents/trees key and is absent from this endpoint's response, so
        the guarded filter dropped every entry and the function returned
        ``([], "")`` — empty list, EMPTY ERROR — for every PR the sweep ever
        scanned. That is exactly the silent degrade-to-empty the paragraph
        above forbids, and it made the ``contracts/<ticket>.yaml`` binding
        signal dead code: every binding came from the PR title alone, and
        ``SKIPPED_AMBIGUOUS_BINDING`` could never fire from a file listing.
        Observed live in run 33050039689 — onex_change_control#7267 bound to
        ``OMN-16682`` on its title while also touching ``contracts/
        OMN-16691.yaml``, i.e. a mis-targeted flip under ``--apply`` rather
        than the conservative skip the guard was written to produce.

        A payload that is a non-empty list but yields zero usable filenames is
        now an ERROR, not an empty success: the only honest reading of "the
        API returned entries this code cannot interpret" is that the listing
        could not be fetched, and the caller must fail closed rather than
        proceed on a binding a real file listing might have contradicted.
        """
        path = f"repos/{repo}/pulls/{number}/files?per_page=100"
        data, error = await self._run_gh_command(
            ["gh", "api", path], gh_timeout_seconds
        )
        if data is None:
            return [], error or f"gh api returned no data for {repo}#{number} files"
        if not isinstance(data, list):
            return [], f"gh api returned non-list files payload for {repo}#{number}"
        files = [
            str(item[_GH_PR_FILE_PATH_KEY])
            for item in data
            if isinstance(item, dict) and item.get(_GH_PR_FILE_PATH_KEY)
        ]
        if data and not files:
            return [], (
                f"gh api returned {len(data)} file entries for {repo}#{number} but "
                f"none carried a {_GH_PR_FILE_PATH_KEY!r} key — refusing to treat an "
                "uninterpretable payload as an empty changed-file list"
            )
        return files, ""

    # -- main entrypoint ---------------------------------------------------

    async def handle(
        self, request: ModelEvidenceAutocloseSweepRequest
    ) -> ModelEvidenceAutocloseSweepResult:
        correlation_id = request.correlation_id or uuid4()

        kill_switch_engaged = self._autoclose_disabled_ctor or bool(
            os.environ.get(_KILL_SWITCH_ENV_VAR, "")
        )
        if kill_switch_engaged:
            logger.warning(
                "%s is set — evidence autoclose sweep disabled, zero I/O performed.",
                _KILL_SWITCH_ENV_VAR,
            )
            return ModelEvidenceAutocloseSweepResult(
                correlation_id=correlation_id,
                dry_run=not request.apply,
                kill_switch_engaged=True,
            )

        since_iso = (
            datetime.now(tz=UTC) - timedelta(hours=request.lookback_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        companions, enum_error = await self._fetch_merged_companions(
            request.occ_repo,
            since_iso,
            request.max_companions,
            request.gh_timeout_seconds,
        )
        if enum_error:
            return ModelEvidenceAutocloseSweepResult(
                correlation_id=correlation_id,
                dry_run=not request.apply,
                success=False,
                error_message=f"GitHub enumeration failed: {enum_error}",
            )

        outcomes: list[ModelEvidenceAutocloseOutcome] = []
        seen_tickets: set[str] = set()
        bindings_extracted = 0

        for pr in companions:
            number = _as_int(pr.get("number"))
            url = str(pr.get("html_url") or "")
            title = str(pr.get("title") or "")
            files, files_error = await self._fetch_pr_files(
                request.occ_repo, number, request.gh_timeout_seconds
            )
            if files_error:
                outcomes.append(
                    ModelEvidenceAutocloseOutcome(
                        companion_pr_number=number,
                        companion_pr_url=url,
                        decision=EnumEvidenceAutocloseDecision.ERROR_GITHUB_API,
                        reason=f"Could not fetch changed-file list: {files_error}",
                    )
                )
                continue
            ticket_id, ambiguous = _extract_ticket_binding(title, files)

            if ambiguous:
                outcomes.append(
                    ModelEvidenceAutocloseOutcome(
                        companion_pr_number=number,
                        companion_pr_url=url,
                        decision=EnumEvidenceAutocloseDecision.SKIPPED_AMBIGUOUS_BINDING,
                        reason=(
                            "More than one distinct OMN-XXXXX ticket id found "
                            "across contract files and title — refusing to guess."
                        ),
                    )
                )
                continue
            if ticket_id is None:
                outcomes.append(
                    ModelEvidenceAutocloseOutcome(
                        companion_pr_number=number,
                        companion_pr_url=url,
                        decision=EnumEvidenceAutocloseDecision.SKIPPED_NO_BINDING,
                        reason="No contracts/OMN-XXXXX.yaml file and no evidence(OMN-XXXXX) title match.",
                    )
                )
                continue

            bindings_extracted += 1
            if ticket_id in seen_tickets:
                # Same ticket bound by more than one companion in this
                # window — already processed by an earlier (more recent)
                # companion in this run; skip the older duplicate.
                continue
            seen_tickets.add(ticket_id)

            outcome = await self._process_ticket(
                ticket_id=ticket_id,
                companion_pr_number=number,
                companion_pr_url=url,
                request=request,
            )
            outcomes.append(outcome)

        flipped = sum(
            1 for o in outcomes if o.decision == EnumEvidenceAutocloseDecision.FLIPPED
        )
        gap_posted = sum(
            1
            for o in outcomes
            # GAP_AC_COVERAGE counts here, not under `errored`: nothing failed,
            # the ticket's evidence base was simply incomplete (OMN-16736).
            if o.decision
            in (
                EnumEvidenceAutocloseDecision.GAP_POSTED,
                EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE,
                # OMN-15911: green, and no green leg proved behavior. Same
                # bucket for the same reason -- the mechanism worked, the
                # evidence was not strong enough to close on.
                EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF,
            )
        )
        skipped = sum(
            1
            for o in outcomes
            if o.decision
            in (
                EnumEvidenceAutocloseDecision.SKIPPED_LABEL,
                # OMN-17891: a caller-asserted refusal. A skip, never a flip
                # and never a gap — the sweep formed no opinion about this
                # ticket's evidence, so counting it anywhere else would report
                # a verdict that was never reached.
                EnumEvidenceAutocloseDecision.SKIPPED_EXCLUDED,
                EnumEvidenceAutocloseDecision.SKIPPED_ALREADY_DONE,
                EnumEvidenceAutocloseDecision.SKIPPED_NO_BINDING,
                EnumEvidenceAutocloseDecision.SKIPPED_AMBIGUOUS_BINDING,
                # OMN-16808: the gap is real and still open, but this run said
                # nothing new. A skip, not a gap post — counting it under
                # `gap_posted` would report comments that were never written.
                EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT,
            )
        )
        errored = sum(
            1
            for o in outcomes
            if o.decision
            in (
                EnumEvidenceAutocloseDecision.ERROR_VERIFY_NONZERO_EXIT,
                EnumEvidenceAutocloseDecision.ERROR_VERIFY_UNPARSEABLE,
                EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                EnumEvidenceAutocloseDecision.ERROR_GITHUB_API,
            )
        )

        return ModelEvidenceAutocloseSweepResult(
            correlation_id=correlation_id,
            dry_run=not request.apply,
            companions_scanned=len(companions),
            bindings_extracted=bindings_extracted,
            tickets_flipped=flipped,
            tickets_gap_posted=gap_posted,
            tickets_skipped=skipped,
            tickets_errored=errored,
            outcomes=tuple(outcomes),
        )

    # -- comment idempotency funnel (OMN-16808) --------------------------

    async def _dedup_gate(
        self, *, issue_id: str, marker: str
    ) -> tuple[Literal["post", "duplicate", "unreadable"], str]:
        """Decide whether this exact statement may be written to this ticket.

        Returns ``(gate, detail)``. ``unreadable`` is the fail-closed verdict:
        the sweep could not establish whether it has already commented, so it
        must not comment. Never returns ``post`` on a read it could not make.
        """
        if not issue_id:
            return "unreadable", "Linear issue payload carried no issue id"
        bodies = await self._linear.fetch_comment_bodies(issue_id)
        if bodies is None:
            return (
                "unreadable",
                "the ticket's existing comments could not be read from Linear",
            )
        if any(marker in body for body in bodies):
            return "duplicate", marker
        return "post", ""

    async def _emit_gap_comment(
        self,
        *,
        base: ModelEvidenceAutocloseOutcome,
        apply: bool,
        issue_id: str,
        marker: str,
        comment_body: str,
    ) -> ModelEvidenceAutocloseOutcome:
        """Single write path for every gap comment (OMN-16808).

        All three gap classes funnel through here so the read-before-write rule
        cannot be satisfied on two paths and forgotten on the third — which is
        exactly how the defect existed: three call sites, three unconditional
        ``create_comment`` calls, no shared gate.

        The dedup read runs in DRY-RUN too. It is a read, so DRY-RUN stays
        zero-write, and the report then previews what ``--apply`` would
        actually do rather than promising a comment the write path would
        suppress.
        """
        gate, detail = await self._dedup_gate(issue_id=issue_id, marker=marker)
        if gate == "unreadable":
            return base.model_copy(
                update={
                    "decision": EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                    "reason": (
                        f"Refusing to comment — {detail}, so the sweep cannot "
                        "establish that it has not already said this here "
                        "(OMN-16808). Fails closed rather than risk a duplicate. "
                        f"Flip withheld regardless: {base.reason}"
                    ),
                }
            )
        if gate == "duplicate":
            return base.model_copy(
                update={
                    "decision": EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT,
                    "reason": (
                        "This exact verdict is already posted on the ticket "
                        f"({marker}) — not repeating it. The gap itself is "
                        f"unchanged and still open: {base.reason}"
                    ),
                }
            )
        if not apply:
            logger.info(
                "[DRY-RUN] Would post %s comment on %s (%s)",
                base.decision.value,
                base.ticket_id,
                base.reason,
            )
            return base.model_copy(update={"reason": f"[DRY-RUN] {base.reason}"})

        commented = await self._linear.create_comment(
            issue_id, f"{comment_body}\n\n{marker}"
        )
        if not commented:
            return base.model_copy(
                update={
                    "decision": EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                    "reason": (
                        "commentCreate mutation failed while posting the gap: "
                        f"{base.reason}"
                    ),
                }
            )
        return base.model_copy(update={"linear_comment_posted": True, "applied": True})

    async def _behavior_proof_outcome(
        self,
        *,
        ticket_id: str,
        companion_pr_number: int,
        companion_pr_url: str,
        apply: bool,
        issue_id: str,
        total_checks: int,
        verified_count: int,
        failed_count: int,
        non_probative_count: int,
    ) -> ModelEvidenceAutocloseOutcome:
        """Withhold the flip: green, but nothing green proved behavior (OMN-15911).

        Mirrors :meth:`_ac_coverage_outcome` exactly — never mutates ticket
        state on any path, and the only write it can make is a comment, under
        ``apply``. The two guards answer different questions and both must
        pass: this one asks what the checks that RAN proved; the AC-coverage
        one asks what the ticket claimed that no check covers.
        """
        reason = (
            f"dod_verify: {verified_count}/{total_checks} checks verified, 0 failed "
            "— and not one of them executed the claimed behavior. Every passing "
            "check binds merge state (a PR is merged) or is a surrogate (a file "
            "exists, a generic suite ran). That is evidence the code landed, not "
            "evidence the system does the thing, so the Done flip is withheld."
        )
        base = ModelEvidenceAutocloseOutcome(
            ticket_id=ticket_id,
            companion_pr_number=companion_pr_number,
            companion_pr_url=companion_pr_url,
            decision=EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF,
            reason=reason,
            dod_verify_total_checks=total_checks,
            dod_verify_verified_count=verified_count,
            dod_verify_failed_count=failed_count,
            dod_verify_non_probative_count=non_probative_count,
            dod_verify_behavior_proving_count=0,
        )
        marker = _sweep_comment_marker(
            EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF,
            (
                str(total_checks),
                str(verified_count),
                str(failed_count),
                "behavior=0",
            ),
        )
        return await self._emit_gap_comment(
            base=base,
            apply=apply,
            issue_id=issue_id,
            marker=marker,
            comment_body=(
                "Proof-class gap (OMN-16106 evidence autoclose sweep) — NOT "
                "flipped.\n\n"
                f"Merged evidence companion: {companion_pr_url}\n"
                f"{reason}\n\n"
                "To make this ticket auto-closable, add at least one check to its "
                "OCC contract that executes the behavior this ticket claims — a "
                "test run, or a run of the product CLI over the changed path. A "
                "`gh pr view --json state` check proves the merge, and the merge "
                "is already established by the companion."
            ),
        )

    async def _ac_coverage_outcome(
        self,
        *,
        ticket_id: str,
        companion_pr_number: int,
        companion_pr_url: str,
        apply: bool,
        issue_id: str,
        ac_gap_reason: str,
        uncovered: tuple[str, ...],
        total_checks: int,
        verified_count: int,
        failed_count: int,
        non_probative_count: int,
        behavior_proving_count: int,
    ) -> ModelEvidenceAutocloseOutcome:
        """Withhold the flip and record/post the AC-coverage gap (OMN-16736).

        Never mutates ticket state on any path -- the only write it can make is
        a comment, and only under ``apply``.
        """
        base = ModelEvidenceAutocloseOutcome(
            ticket_id=ticket_id,
            companion_pr_number=companion_pr_number,
            companion_pr_url=companion_pr_url,
            decision=EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE,
            reason=ac_gap_reason,
            dod_verify_total_checks=total_checks,
            dod_verify_verified_count=verified_count,
            dod_verify_failed_count=failed_count,
            dod_verify_non_probative_count=non_probative_count,
            dod_verify_behavior_proving_count=behavior_proving_count,
            uncovered_acceptance_criteria=uncovered,
        )
        # The uncovered criteria are part of the statement, so they are part of
        # the fingerprint: a description whose unchecked boxes changed is a
        # different gap and does get a fresh comment.
        marker = _sweep_comment_marker(
            EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE,
            (
                str(total_checks),
                str(verified_count),
                str(failed_count),
                f"behavior={behavior_proving_count}",
                *uncovered,
            ),
        )
        return await self._emit_gap_comment(
            base=base,
            apply=apply,
            issue_id=issue_id,
            marker=marker,
            comment_body=(
                "AC-coverage gap (OMN-16106 evidence autoclose sweep) — NOT flipped.\n\n"
                f"Merged evidence companion: {companion_pr_url}\n"
                f"dod_verify: {verified_count}/{total_checks} ACs verified, "
                f"{failed_count} failed — clean, but it verifies only the checks "
                "declared in the OCC contract.\n\n"
                f"{ac_gap_reason}\n\n"
                "Acceptance criteria found in this description:\n"
                f"{_format_uncovered(uncovered)}\n\n"
                "To make this ticket auto-closable, encode these as checks in "
                "its OCC contract (or check the boxes off if they are genuinely "
                "done) — the sweep will flip it on the next run."
            ),
        )

    async def _process_ticket(
        self,
        *,
        ticket_id: str,
        companion_pr_number: int,
        companion_pr_url: str,
        request: ModelEvidenceAutocloseSweepRequest,
    ) -> ModelEvidenceAutocloseOutcome:
        # OMN-17891. The caller-asserted fence, and it is FIRST -- ahead of the
        # Linear read, the label gate, the state gate and the dod_verify
        # subprocess. An excluded candidate therefore costs zero I/O, which is
        # what makes the refusal terminal: no transport failure, no verdict and
        # no later branch can reclassify it, so SKIPPED_EXCLUDED in the record
        # always means the fence applied rather than that it was reached.
        #
        # Match is case-insensitive and whitespace-stripped on BOTH sides. The
        # value arrives typed into a workflow_dispatch box or spliced from a
        # script's output, so `omn-17857` and a trailing space are the expected
        # shapes; a near-miss here does not fail loudly, it flips the ticket the
        # operator was fencing off.
        excluded = {
            candidate.strip().upper()
            for candidate in request.exclude_tickets
            if candidate.strip()
        }
        if ticket_id.strip().upper() in excluded:
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=EnumEvidenceAutocloseDecision.SKIPPED_EXCLUDED,
                reason=(
                    f"{ticket_id} is on the caller-supplied exclusion list — "
                    "refused before any Linear read, dod_verify never ran, and "
                    "no verdict was reached about this ticket by this run."
                ),
            )

        issue = await self._linear.fetch_issue(ticket_id)
        if issue is None:
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                reason="Failed to fetch ticket state/labels from Linear.",
            )

        labels_conn = issue.get("labels")
        label_nodes = labels_conn.get("nodes") if isinstance(labels_conn, dict) else []
        label_names = {
            str(node.get("name", "")).strip().lower()
            for node in (label_nodes or [])
            if isinstance(node, dict)
        }
        if request.close_if_done_label.strip().lower() in label_names:
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=EnumEvidenceAutocloseDecision.SKIPPED_LABEL,
                reason=f"Ticket carries '{request.close_if_done_label}' — manual decision-only path.",
            )

        state = issue.get("state")
        state_type = str(state.get("type", "")) if isinstance(state, dict) else ""
        if state_type in ("completed", "canceled"):
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=EnumEvidenceAutocloseDecision.SKIPPED_ALREADY_DONE,
                reason=f"Ticket state type is already '{state_type}'.",
            )

        dod_result, exit_code, verify_error = await self._run_dod_verify_command(
            ticket_id, request.dispatch_cwd, request.dod_verify_timeout_seconds
        )
        if dod_result is None:
            decision = (
                EnumEvidenceAutocloseDecision.ERROR_VERIFY_UNPARSEABLE
                if exit_code == 0
                else EnumEvidenceAutocloseDecision.ERROR_VERIFY_NONZERO_EXIT
            )
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=decision,
                reason=f"dod_verify exit_code={exit_code}: {verify_error}",
            )

        # Read the verdict from whichever arm the receipt DECLARES it is
        # carrying (OMN-16961). A refusal here always names its own cause —
        # undeclared shape, drifted `result_model`, or a genuinely unreached
        # verdict — so "the sweep could not read it" is never confusable with
        # "the ticket is not proven". Absent counts still fail closed: nothing
        # is coerced to 0 and nothing unread is ever counted as proof.
        verdict, verdict_refusal = _extract_dod_verify_verdict(dod_result)
        if verdict is None:
            decision = (
                EnumEvidenceAutocloseDecision.ERROR_VERIFY_UNPARSEABLE
                if exit_code == 0
                else EnumEvidenceAutocloseDecision.ERROR_VERIFY_NONZERO_EXIT
            )
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=decision,
                reason=f"dod_verify exit_code={exit_code}: {verdict_refusal}",
            )

        total_checks = _as_int(verdict.get("total_checks"))
        verified_count = _as_int(verdict.get("verified_count"))
        failed_count = _as_int(verdict.get("failed_count"))
        non_probative_count = _as_int(verdict.get(_DOD_VERIFY_NON_PROBATIVE_KEY))
        verify_status = str(verdict.get("status") or "").strip().lower()
        # OMN-16905: read the behaviour count HERE, with every other counter,
        # not inside the `all_verified` branch below. It is only the flip
        # predicate that needs it conditionally; the OUTCOME row needs it
        # always, because that row is the machine-readable record the fleet is
        # triaged from.
        #
        # Reading it only under `all_verified` left `dod_verify_behavior_proving
        # _count` at its model default of 0 on every gap/skip path, so the row
        # could not distinguish "no behaviour proof was declared" from
        # "behaviour proof ran, passed, and something else held the flip". In
        # run 33210163405 that produced a 1-vs-0 split against the diagnose
        # leg's own dump for OMN-16803 with all four other counters identical,
        # and the split was read as a classifier regression between the gate
        # venv and the OMN-16846 dispatch venv. There was no such regression:
        # OMN-16803's verdict was `status=skipped`, it took the gap path, and
        # the 0 was structural. Reporting the measured value on every path is
        # what stops that misreading recurring.
        #
        # Absent key stays 0 and is handled as ERROR_VERIFY_UNPARSEABLE below
        # (OMN-15911) — this line never infers a value the verifier did not give.
        behavior_proving_count = _as_int(verdict.get(_DOD_VERIFY_BEHAVIOR_KEY))

        # Both dod_verify's OWN terminal status and the arithmetic must agree.
        # The arithmetic is the stricter of the two: dod_verify reports VERIFIED
        # when *some* checks were skipped (as long as not all of them were), and
        # a skipped check is not proof of anything.
        #
        # OMN-16821: the denominator has to be the set of checks that could
        # have carried a PRODUCT-DEPENDENT verdict. `total_checks` already
        # excludes `superseded` upstream on that reasoning (OMN-15390);
        # `non_probative` is the same class of entry one axis over — it did
        # execute and it did exit 0, but its exit status could not have gone
        # the other way for a product reason, so requiring it to appear in
        # `verified_count` demanded a proof it is definitionally incapable of
        # supplying. Adding it back on the left is what makes the equality
        # satisfiable at all; every other conjunct is the strictness that
        # keeps it honest:
        #   * `failed_count == 0`     — a real red is still a gap.
        #   * `verified_count > 0`    — an ALL-non-probative contract cannot
        #     satisfy the equality by arithmetic alone, independent of
        #     dod_verify's own `skipped` status for that shape. Two guards,
        #     because the merge-state-only corpus is the exact population this
        #     mechanism must never flip.
        #   * the equality itself     — a `skipped` check is neither verified
        #     nor non-probative, so it still breaks it. A check that never ran
        #     proves nothing.
        all_verified = (
            verify_status == _DOD_VERIFY_STATUS_VERIFIED
            and total_checks > 0
            and failed_count == 0
            and verified_count > 0
            and verified_count + non_probative_count == total_checks
        )

        issue_id = str(issue.get("id") or "")
        team = issue.get("team")
        team_id = str(team.get("id") or "") if isinstance(team, dict) else ""

        if all_verified:
            # OMN-15911: green is necessary, not sufficient — and the FIRST
            # question is what the green legs proved, not what the ticket body
            # says. `verified_count == total_checks` is satisfiable entirely by
            # `gh pr view --json state` reads, which bind merge state and say
            # nothing about whether the merged code works.
            if _DOD_VERIFY_BEHAVIOR_KEY not in verdict:
                return ModelEvidenceAutocloseOutcome(
                    ticket_id=ticket_id,
                    companion_pr_number=companion_pr_number,
                    companion_pr_url=companion_pr_url,
                    decision=EnumEvidenceAutocloseDecision.ERROR_VERIFY_UNPARSEABLE,
                    reason=(
                        f"dod_verify reported no `result.{_DOD_VERIFY_VERDICT_KEY}"
                        f".{_DOD_VERIFY_BEHAVIOR_KEY}` — the verifier predates "
                        "OMN-15911 and cannot say whether any green check proved "
                        "behavior. Refusing to infer it in either direction."
                    ),
                    dod_verify_total_checks=total_checks,
                    dod_verify_verified_count=verified_count,
                    dod_verify_failed_count=failed_count,
                    dod_verify_non_probative_count=non_probative_count,
                    # OMN-16905: explicit, not the model default. This is the
                    # one branch where 0 does NOT mean "measured zero" — the
                    # key is absent, which is exactly why this branch refuses
                    # to decide. Writing it out keeps the "report every counter
                    # you read" invariant mechanically checkable (the AST gate
                    # in test_handler_evidence_autoclose_sweep.py) instead of
                    # leaving one silent hole in it.
                    dod_verify_behavior_proving_count=0,
                )
            if behavior_proving_count <= 0:
                return await self._behavior_proof_outcome(
                    ticket_id=ticket_id,
                    companion_pr_number=companion_pr_number,
                    companion_pr_url=companion_pr_url,
                    apply=request.apply,
                    issue_id=issue_id,
                    total_checks=total_checks,
                    verified_count=verified_count,
                    failed_count=failed_count,
                    non_probative_count=non_probative_count,
                )

            # OMN-16736: dod_verify being green is necessary, not sufficient.
            # Re-read the ticket body for criteria its checks never covered
            # BEFORE any flip path (dry-run included, so a DRY-RUN report is
            # an honest preview of what --apply would do).
            description_raw = issue.get("description")
            description = description_raw if isinstance(description_raw, str) else ""
            ac_gap_reason, uncovered = _ac_coverage_gap(description, total_checks)
            if ac_gap_reason:
                return await self._ac_coverage_outcome(
                    ticket_id=ticket_id,
                    companion_pr_number=companion_pr_number,
                    companion_pr_url=companion_pr_url,
                    apply=request.apply,
                    issue_id=issue_id,
                    ac_gap_reason=ac_gap_reason,
                    uncovered=uncovered,
                    total_checks=total_checks,
                    verified_count=verified_count,
                    failed_count=failed_count,
                    non_probative_count=non_probative_count,
                    behavior_proving_count=behavior_proving_count,
                )

            # OMN-16821: `non_probative_count` is stated because without it a
            # legitimate flip reads as an unexplained shortfall — "6/12 ACs
            # verified" alone looks like half the contract went unproven, when
            # the other six were provenance entries incapable of carrying a
            # product-dependent verdict. The equality that released the flip is
            # `verified + non_probative == total`, so the reason has to show
            # both terms or it cannot be checked by whoever reads it.
            reason = (
                f"dod_verify: {verified_count}/{total_checks} ACs verified "
                f"({non_probative_count} non-probative), "
                f"0 failed, {behavior_proving_count} behavior-proving. "
                f"Companion: {companion_pr_url}"
            )
            if not request.apply:
                logger.info("[DRY-RUN] Would flip %s to Done (%s)", ticket_id, reason)
                return ModelEvidenceAutocloseOutcome(
                    ticket_id=ticket_id,
                    companion_pr_number=companion_pr_number,
                    companion_pr_url=companion_pr_url,
                    decision=EnumEvidenceAutocloseDecision.FLIPPED,
                    reason=f"[DRY-RUN] {reason}",
                    dod_verify_total_checks=total_checks,
                    dod_verify_verified_count=verified_count,
                    dod_verify_failed_count=failed_count,
                    dod_verify_non_probative_count=non_probative_count,
                    dod_verify_behavior_proving_count=behavior_proving_count,
                    applied=False,
                )
            if not issue_id or not team_id:
                return ModelEvidenceAutocloseOutcome(
                    ticket_id=ticket_id,
                    companion_pr_number=companion_pr_number,
                    companion_pr_url=companion_pr_url,
                    decision=EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                    reason="Linear issue payload missing id/team — refusing to flip.",
                    dod_verify_total_checks=total_checks,
                    dod_verify_verified_count=verified_count,
                    dod_verify_failed_count=failed_count,
                    dod_verify_non_probative_count=non_probative_count,
                    dod_verify_behavior_proving_count=behavior_proving_count,
                )
            done_state_id = await self._linear.fetch_done_state_id(team_id)
            if not done_state_id:
                return ModelEvidenceAutocloseOutcome(
                    ticket_id=ticket_id,
                    companion_pr_number=companion_pr_number,
                    companion_pr_url=companion_pr_url,
                    decision=EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                    reason="Could not resolve team's completed-type 'Done' state.",
                    dod_verify_total_checks=total_checks,
                    dod_verify_verified_count=verified_count,
                    dod_verify_failed_count=failed_count,
                    dod_verify_non_probative_count=non_probative_count,
                    dod_verify_behavior_proving_count=behavior_proving_count,
                )
            flipped = await self._linear.update_issue_state(issue_id, done_state_id)
            if not flipped:
                return ModelEvidenceAutocloseOutcome(
                    ticket_id=ticket_id,
                    companion_pr_number=companion_pr_number,
                    companion_pr_url=companion_pr_url,
                    decision=EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                    reason="issueUpdate(stateId) mutation failed.",
                    dod_verify_total_checks=total_checks,
                    dod_verify_verified_count=verified_count,
                    dod_verify_failed_count=failed_count,
                    dod_verify_non_probative_count=non_probative_count,
                    dod_verify_behavior_proving_count=behavior_proving_count,
                )
            # No dedup gate on the flip audit comment (OMN-16808), and that is
            # deliberate: this line is reached only after `issueUpdate` has
            # already moved the ticket into a completed state, which the next
            # run short-circuits at SKIPPED_ALREADY_DONE before any verdict is
            # reached. The flip path is self-limiting by its own terminal
            # state; the gap paths have no such terminal state, which is why
            # they need the read-before-write gate and this does not.
            commented = await self._linear.create_comment(
                issue_id,
                (
                    "Automatic Done flip (OMN-16106 evidence autoclose sweep).\n\n"
                    f"Merged evidence companion: {companion_pr_url}\n"
                    f"Behavior-proving checks: {behavior_proving_count} "
                    "(OMN-15911 — at least one check executed the claimed "
                    "behavior, not only a merge-state read).\n"
                    f"dod_verify: {verified_count}/{total_checks} ACs verified "
                    f"({non_probative_count} non-probative), {failed_count} failed."
                ),
            )
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=EnumEvidenceAutocloseDecision.FLIPPED,
                reason=reason,
                dod_verify_total_checks=total_checks,
                dod_verify_verified_count=verified_count,
                dod_verify_failed_count=failed_count,
                dod_verify_non_probative_count=non_probative_count,
                dod_verify_behavior_proving_count=behavior_proving_count,
                linear_comment_posted=commented,
                applied=True,
            )

        # Gap path.
        shortfall = _gap_shortfall(
            verify_status=verify_status,
            total_checks=total_checks,
            verified_count=verified_count,
            failed_count=failed_count,
            non_probative_count=non_probative_count,
        )
        gap_reason = (
            f"dod_verify: {verified_count}/{total_checks} ACs verified, "
            f"{failed_count} failed — {shortfall}"
        )
        gap_base = ModelEvidenceAutocloseOutcome(
            ticket_id=ticket_id,
            companion_pr_number=companion_pr_number,
            companion_pr_url=companion_pr_url,
            decision=EnumEvidenceAutocloseDecision.GAP_POSTED,
            reason=gap_reason,
            dod_verify_total_checks=total_checks,
            dod_verify_verified_count=verified_count,
            dod_verify_failed_count=failed_count,
            dod_verify_non_probative_count=non_probative_count,
            # OMN-16905: the measured value, not the model default. This is the
            # path OMN-16803 took, and the default is what made its row read 0
            # while its own verdict said 1.
            dod_verify_behavior_proving_count=behavior_proving_count,
        )
        # OMN-16821: `non_probative_count` joins the OMN-16808 dedup
        # fingerprint because the STATEMENT now varies with it. Two verdicts
        # sharing (total, verified, failed) and differing in non-probative
        # count produce different `shortfall` wording, so keying without it
        # would let a stale comment suppress the corrected one — the dedup
        # gate must track what was said, not merely which counts were seen.
        marker = _sweep_comment_marker(
            EnumEvidenceAutocloseDecision.GAP_POSTED,
            (
                str(total_checks),
                str(verified_count),
                str(failed_count),
                str(non_probative_count),
            ),
        )
        return await self._emit_gap_comment(
            base=gap_base,
            apply=request.apply,
            issue_id=issue_id,
            marker=marker,
            comment_body=(
                "Evidence gap (OMN-16106 evidence autoclose sweep) — NOT flipped.\n\n"
                f"Merged evidence companion: {companion_pr_url}\n"
                f"dod_verify: {verified_count}/{total_checks} ACs verified, "
                f"{failed_count} failed — {shortfall}"
            ),
        )
