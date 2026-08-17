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
   ``uv run onex skill dod_verify <ticket-id>``. A non-zero exit or
   unparseable output fails closed (recorded, never flips). A zero exit
   with the AC counts read from the printed ``ModelSkillResult`` JSON
   decides FLIP (all ACs verified, zero failed) vs GAP (anything else).
6. ``apply=False`` (the default) performs every read above but never calls
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

# Evidence-Ticket binding patterns.
_CONTRACT_FILE_RE = re.compile(r"^contracts/(OMN-\d+)\.yaml$")
_TITLE_EVIDENCE_RE = re.compile(r"evidence\((OMN-\d+)\)", re.IGNORECASE)

_LINEAR_API_URL = "https://api.linear.app/graphql"  # url-authority-ok: fixed public GraphQL API, no ONEX routing authority

_ISSUE_QUERY = """
query GetIssue($id: String!) {
  issue(id: $id) {
    id
    identifier
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
        gh_timeout_seconds: float = 30.0,
        autoclose_disabled: bool | None = None,
        run_gh_command: TypeRunGhCommand | None = None,
        run_dod_verify_command: TypeRunDodVerifyCommand | None = None,
    ) -> None:
        # ``autoclose_disabled`` mirrors GitHubTransport's env-fallback-in-
        # __init__ precedent: read at construction time, override injectable
        # for tests. Re-checked defensively at the top of handle() too so a
        # zero-arg contract-driven construction can never silently skip it.
        self._linear = linear_client if linear_client is not None else _LinearClient()
        self._gh_timeout_seconds = gh_timeout_seconds
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
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
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
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd or None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            await _reap_timed_out_process(proc)
            return None, -1, f"Timeout running dod_verify for {ticket_id}"
        except OSError as exc:
            await _reap_timed_out_process(proc)
            return None, -1, f"OS error running dod_verify for {ticket_id}: {exc}"
        exit_code = proc.returncode if proc.returncode is not None else -1
        if exit_code != 0:
            return None, exit_code, stderr.decode(errors="replace").strip()
        try:
            parsed = json.loads(stdout.decode())
        except json.JSONDecodeError as exc:
            return None, exit_code, f"Invalid dod_verify JSON: {exc}"
        if not isinstance(parsed, dict):
            return None, exit_code, "dod_verify output was not a JSON object"
        return parsed, exit_code, ""

    # -- GitHub enumeration ----------------------------------------------

    async def _fetch_merged_companions(
        self, repo: str, since_iso: str, max_companions: int
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
                ["gh", "api", path], self._gh_timeout_seconds
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

    async def _fetch_pr_files(self, repo: str, number: int) -> tuple[list[str], str]:
        """Fetch a PR's changed-file paths. Never silently degrades to empty.

        Returns ``(files, error)``. On a genuine fetch failure ``error`` is
        non-empty and ``files`` is empty — callers MUST treat that as
        ERROR_GITHUB_API, never as "this companion touched zero files" (which
        would fall through to a title-only binding match a real file listing
        might have disambiguated or contradicted).
        """
        path = f"repos/{repo}/pulls/{number}/files?per_page=100"
        data, error = await self._run_gh_command(
            ["gh", "api", path], self._gh_timeout_seconds
        )
        if data is None:
            return [], error or f"gh api returned no data for {repo}#{number} files"
        if not isinstance(data, list):
            return [], f"gh api returned non-list files payload for {repo}#{number}"
        return [
            str(item["path"])
            for item in data
            if isinstance(item, dict) and item.get("path")
        ], ""

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
            request.occ_repo, since_iso, request.max_companions
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
            files, files_error = await self._fetch_pr_files(request.occ_repo, number)
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
            if o.decision == EnumEvidenceAutocloseDecision.GAP_POSTED
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

        verify_payload = dod_result.get("result")
        verify_payload = verify_payload if isinstance(verify_payload, dict) else {}
        total_checks = _as_int(verify_payload.get("total_checks"))
        verified_count = _as_int(verify_payload.get("verified_count"))
        failed_count = _as_int(verify_payload.get("failed_count"))

        all_verified = (
            total_checks > 0 and failed_count == 0 and verified_count == total_checks
        )

        issue_id = str(issue.get("id") or "")
        team = issue.get("team")
        team_id = str(team.get("id") or "") if isinstance(team, dict) else ""

        if all_verified:
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
