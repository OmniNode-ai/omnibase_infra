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
5. Otherwise run the EXISTING verifier exactly as the controller does:
   ``uv run onex skill dod_verify <ticket-id>``. Its stdout is parsed
   REGARDLESS of exit code — the CLI exits 1 on every genuine evidence gap
   while still printing a complete ``ModelSkillResult`` — and the verdict is
   read from ``result.terminal_payload`` (``result`` itself carries only the
   dispatch outcome). FLIP requires all of: the verifier's own
   ``status == "verified"``, ``total_checks > 0``, zero failed, and
   ``verified_count == total_checks``. Anything else that still reached a
   verdict is a GAP. Output with no parseable verdict in it fails closed as
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
7. ``apply=False`` (the default) performs every read above but never calls
   a Linear mutation — every decision is logged as "would-do". ``apply=True``
   performs the real ``issueUpdate``/``commentCreate`` mutation.

Non-blocking Design
--------------------
Per-ticket failures (Linear API errors, dod_verify crashes) are recorded in
the outcome list and do not abort the sweep — only a sweep-level failure
(GitHub enumeration itself failing) sets ``result.success = False``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
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

# Injectable subprocess-runner signatures (real impls call `gh`/`uv run onex`;
# tests inject fakes with the same shape).
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
# The printed ModelSkillResult nests the node's own terminal state one level
# below `result`: `result` carries the DISPATCH outcome (workflow_result,
# exit_code, workflow, handler_result, error, capture_log) and
# `result.terminal_payload` carries the VERIFICATION outcome (status,
# total_checks, verified_count, failed_count, skipped_count,
# superseded_count). `result.total_checks` does not exist — verified against a
# live capture, committed at tests/fixtures/omn16736/ (OMN-16736).
_DOD_VERIFY_VERDICT_KEY = "terminal_payload"
# dod_verify's own terminal status for "every verdict-bearing check passed".
# `EnumDodVerifyStatus.VERIFIED` in omnimarket.
_DOD_VERIFY_STATUS_VERIFIED = "verified"

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
        args = ["uv", "run", "onex", "skill", "dod_verify", ticket_id]
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
            )
        )
        skipped = sum(
            1
            for o in outcomes
            if o.decision
            in (
                EnumEvidenceAutocloseDecision.SKIPPED_LABEL,
                EnumEvidenceAutocloseDecision.SKIPPED_ALREADY_DONE,
                EnumEvidenceAutocloseDecision.SKIPPED_NO_BINDING,
                EnumEvidenceAutocloseDecision.SKIPPED_AMBIGUOUS_BINDING,
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
            uncovered_acceptance_criteria=uncovered,
        )
        if not apply:
            logger.info(
                "[DRY-RUN] Would withhold the Done flip on %s (%s)",
                ticket_id,
                ac_gap_reason,
            )
            return base.model_copy(update={"reason": f"[DRY-RUN] {ac_gap_reason}"})

        if not issue_id:
            # A missing issue id cannot be commented on -- but the flip is
            # already withheld, so this stays a Linear-API error rather than
            # silently degrading into "no gap found".
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                reason=(
                    "Linear issue payload missing id — refusing to comment. "
                    f"Flip withheld regardless: {ac_gap_reason}"
                ),
                dod_verify_total_checks=total_checks,
                dod_verify_verified_count=verified_count,
                dod_verify_failed_count=failed_count,
                uncovered_acceptance_criteria=uncovered,
            )

        commented = await self._linear.create_comment(
            issue_id,
            (
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
        return base.model_copy(
            update={"linear_comment_posted": commented, "applied": True}
        )

    async def _process_ticket(
        self,
        *,
        ticket_id: str,
        companion_pr_number: int,
        companion_pr_url: str,
        request: ModelEvidenceAutocloseSweepRequest,
    ) -> ModelEvidenceAutocloseOutcome:
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

        dispatch_payload = dod_result.get("result")
        dispatch_payload = (
            dispatch_payload if isinstance(dispatch_payload, dict) else {}
        )
        verdict = dispatch_payload.get(_DOD_VERIFY_VERDICT_KEY)
        verdict = verdict if isinstance(verdict, dict) else {}

        # A ModelSkillResult with no terminal payload means the dispatch
        # produced output but the verifier never reached a verdict. Absent the
        # counts there is nothing to decide on, so fail closed rather than
        # letting `_as_int(None) -> 0` silently manufacture a 0/0 "gap" that
        # reads like a ticket problem (OMN-16736).
        if "total_checks" not in verdict:
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
                reason=(
                    f"dod_verify exit_code={exit_code} produced JSON with no "
                    f"`result.{_DOD_VERIFY_VERDICT_KEY}.total_checks` — no "
                    "verdict was reached, so no flip or gap can be inferred."
                ),
            )

        total_checks = _as_int(verdict.get("total_checks"))
        verified_count = _as_int(verdict.get("verified_count"))
        failed_count = _as_int(verdict.get("failed_count"))
        verify_status = str(verdict.get("status") or "").strip().lower()

        # Both dod_verify's OWN terminal status and the arithmetic must agree.
        # The arithmetic is the stricter of the two: dod_verify reports VERIFIED
        # when *some* checks were skipped (as long as not all of them were), and
        # a skipped check is not proof of anything. `total_checks` is already the
        # verdict-bearing denominator — superseded entries are excluded from it
        # upstream (OMN-15390) — so `verified_count == total_checks` is the
        # honest "every check that could carry a verdict, did".
        all_verified = (
            verify_status == _DOD_VERIFY_STATUS_VERIFIED
            and total_checks > 0
            and failed_count == 0
            and verified_count == total_checks
        )

        issue_id = str(issue.get("id") or "")
        team = issue.get("team")
        team_id = str(team.get("id") or "") if isinstance(team, dict) else ""

        if all_verified:
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
                )

            reason = (
                f"dod_verify: {verified_count}/{total_checks} ACs verified, "
                f"0 failed. Companion: {companion_pr_url}"
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
                )
            commented = await self._linear.create_comment(
                issue_id,
                (
                    "Automatic Done flip (OMN-16106 evidence autoclose sweep).\n\n"
                    f"Merged evidence companion: {companion_pr_url}\n"
                    f"dod_verify: {verified_count}/{total_checks} ACs verified, "
                    f"{failed_count} failed."
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
                linear_comment_posted=commented,
                applied=True,
            )

        # Gap path.
        gap_reason = (
            f"dod_verify: {verified_count}/{total_checks} ACs verified, "
            f"{failed_count} failed — not all ACs are receipt-proven."
        )
        if not request.apply:
            logger.info(
                "[DRY-RUN] Would post gap comment on %s (%s)", ticket_id, gap_reason
            )
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=EnumEvidenceAutocloseDecision.GAP_POSTED,
                reason=f"[DRY-RUN] {gap_reason}",
                dod_verify_total_checks=total_checks,
                dod_verify_verified_count=verified_count,
                dod_verify_failed_count=failed_count,
                applied=False,
            )
        if not issue_id:
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                reason="Linear issue payload missing id — refusing to comment.",
                dod_verify_total_checks=total_checks,
                dod_verify_verified_count=verified_count,
                dod_verify_failed_count=failed_count,
            )
        commented = await self._linear.create_comment(
            issue_id,
            (
                "Evidence gap (OMN-16106 evidence autoclose sweep) — NOT flipped.\n\n"
                f"Merged evidence companion: {companion_pr_url}\n"
                f"dod_verify: {verified_count}/{total_checks} ACs verified, "
                f"{failed_count} failed."
            ),
        )
        if not commented:
            return ModelEvidenceAutocloseOutcome(
                ticket_id=ticket_id,
                companion_pr_number=companion_pr_number,
                companion_pr_url=companion_pr_url,
                decision=EnumEvidenceAutocloseDecision.ERROR_LINEAR_API,
                reason="commentCreate mutation failed while posting the gap.",
                dod_verify_total_checks=total_checks,
                dod_verify_verified_count=verified_count,
                dod_verify_failed_count=failed_count,
            )
        return ModelEvidenceAutocloseOutcome(
            ticket_id=ticket_id,
            companion_pr_number=companion_pr_number,
            companion_pr_url=companion_pr_url,
            decision=EnumEvidenceAutocloseDecision.GAP_POSTED,
            reason=gap_reason,
            dod_verify_total_checks=total_checks,
            dod_verify_verified_count=verified_count,
            dod_verify_failed_count=failed_count,
            linear_comment_posted=True,
            applied=True,
        )
