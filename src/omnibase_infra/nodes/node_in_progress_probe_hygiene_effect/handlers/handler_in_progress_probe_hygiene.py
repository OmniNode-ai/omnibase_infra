# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for the in-progress probe hygiene sweep (OMN-17942).

Why this exists
---------------
The scheduled evidence closer (OMN-16106) re-runs ``onex skill dod_verify``
against the checks a ticket's OCC contract declares. A ticket that declares
none is not merely *failing* that verifier — it is **invisible** to it, and to
every other mechanical closing path. It can only ever be closed by a person
reading it, and nothing on the board says so.

Four tickets in the 2026-08-31 sprint are in exactly that state (OMN-17926,
OMN-17277, OMN-17380, OMN-17353): prose deliverables — a PRD, a doctrine
paragraph, a validator hand-off — with no executable assertion anywhere. They
do not appear in the closer's outcomes at all, because the closer enumerates
merged OCC companions and they have none. Their absence reads exactly like
"nothing to report".

``OMN-17942``'s creation gate stops NEW tickets entering In Progress without a
probe line. This sweep is the other half: it names the ones already there.

What it does, and what it must never do
---------------------------------------
It **comments**. It never touches state, never sets a label, and holds no flip
budget. A ticket with no probe is not wrong — it may be genuinely un-probeable
work — and the finding is that nothing mechanical can close it, which is a fact
its owner needs and not a verdict the sweep is entitled to act on.

Idempotency is by marker line, the same construction as the closer's OMN-16808
dedup: the sweep says a thing ONCE per ticket. A comment history it cannot read
fails closed as an error rather than risking a duplicate, because "there is no
prior comment" and "I could not look" are different facts and only one of them
releases a write.

Where the probe can live
------------------------
Two places, checked in this order:

1. The ticket's OCC contract ``contracts/<TICKET>.yaml`` — any
   ``dod_evidence[].checks[]`` entry. This is the one the closer actually
   runs, so it is the authoritative answer.
2. A ``Probe: <command> => <observation>`` line in the Linear description —
   the OMN-17942 creation-gate grammar, matched whole-line and unbulleted for
   the reason CLAUDE.md rule 15 gives.

The second is a second chance, not the only one: every ticket filed before the
creation gate landed has no probe line and many of them have perfectly good
contract checks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.enum_probe_hygiene_decision import (
    EnumProbeHygieneDecision,
)
from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.model_probe_hygiene_outcome import (
    ModelProbeHygieneOutcome,
)
from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.model_probe_hygiene_request import (
    ModelProbeHygieneRequest,
)
from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.model_probe_hygiene_result import (
    ModelProbeHygieneResult,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

_LINEAR_API_URL: Final[str] = (
    "https://api.linear.app/graphql"  # url-authority-ok: fixed public GraphQL API, no ONEX routing authority
)

#: The marker that makes this sweep's comment identifiable by a LATER run. It
#: must be stable across every wording change below it, because it is the only
#: thing standing between one honest finding and a comment per tick forever.
_HYGIENE_COMMENT_MARKER: Final[str] = (
    "<!-- onex:in-progress-probe-hygiene:no-executable-probe -->"
)

#: ``Probe: <command> => <observation>``, whole-line and unbulleted — the same
#: grammar the OMN-17942 creation gate admits, so a ticket that satisfied the
#: gate at filing time satisfies this sweep without being edited.
_PROBE_LINE: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*Probe:[ \t]*(?P<probe>.*?)[ \t]*$", re.MULTILINE
)
_PROBE_SPLIT: Final[str] = "=>"

_TICKET_ID: Final[re.Pattern[str]] = re.compile(r"^OMN-\d+$", re.IGNORECASE)

_HTTP_TOO_MANY_REQUESTS: Final[int] = 429
_HTTP_SERVER_ERROR: Final[int] = 500
_RETRY_MAX_DELAY_S: Final[float] = 30.0
_RATE_LIMIT_TOKENS: Final[tuple[str, ...]] = (
    "ratelimit",
    "rate limit",
    "too many requests",
)
_MAX_COMMENT_PAGES: Final[int] = 5

_IN_PROGRESS_QUERY: Final[str] = """
query InProgressIssues($first: Int!, $after: String, $filter: IssueFilter) {
  issues(first: $first, after: $after, filter: $filter) {
    nodes {
      id
      identifier
      title
      description
      state { name type }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

_ISSUE_COMMENTS_QUERY: Final[str] = """
query IssueComments($id: String!, $after: String) {
  issue(id: $id) {
    comments(first: 100, after: $after) {
      nodes { body }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

_COMMENT_CREATE_MUTATION: Final[str] = """
mutation CreateComment($issueId: String!, $body: String!) {
  commentCreate(input: { issueId: $issueId, body: $body }) {
    success
  }
}
"""


def _well_formed_probe_lines(description: str) -> int:
    """Count description lines carrying a well-formed probe.

    Both halves are required. A command with no expected observation can only
    be adjudicated by a person reading its output, which is the state this
    sweep exists to surface — counting it would report the problem as solved.
    """
    count = 0
    for match in _PROBE_LINE.finditer(description):
        candidate = match.group("probe").strip()
        head, separator, tail = candidate.partition(_PROBE_SPLIT)
        if separator and head.strip() and tail.strip():
            count += 1
    return count


def _occ_contract_check_count(occ_repo_dir: str, ticket: str) -> int | None:
    """Executable checks declared in the ticket's OCC contract.

    ``None`` means the contract could NOT be read — no clone, an unreadable
    path, malformed YAML. ``0`` means the contract was read and declares
    nothing. Those are different facts with the same cardinality, and
    collapsing them would report a broken runner as a board-wide finding.

    ``ticket`` is validated against the ticket-id shape before it is joined
    onto a path: it arrives from the Linear API, and a path segment built from
    remote text is a traversal waiting to happen.
    """
    if not occ_repo_dir or not _TICKET_ID.match(ticket):
        return None
    contract = Path(occ_repo_dir) / "contracts" / f"{ticket.upper()}.yaml"
    try:
        raw = yaml.safe_load(contract.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # The one absence that is a real answer: the governance clone is
        # readable and this ticket simply has no contract in it.
        return 0 if Path(occ_repo_dir, "contracts").is_dir() else None
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "OCC contract for %s unreadable: %s", ticket, sanitize_error_message(exc)
        )
        return None
    if not isinstance(raw, dict):
        return None
    evidence = raw.get("dod_evidence")
    if not isinstance(evidence, list):
        return 0
    total = 0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        checks = item.get("checks")
        if isinstance(checks, list):
            total += sum(1 for check in checks if isinstance(check, dict))
    return total


def _comment_body(ticket: str) -> str:
    """The one comment this sweep ever writes about a ticket."""
    return (
        f"{_HYGIENE_COMMENT_MARKER}\n"
        "**This ticket is In Progress and declares no executable close probe.**\n\n"
        "The scheduled evidence closer (OMN-16106) closes a ticket by re-running "
        "`onex skill dod_verify`, which runs the checks the ticket's OCC contract "
        f"declares. `contracts/{ticket}.yaml` declares none, and the "
        "description carries no probe line — so no mechanical path can ever close "
        "this ticket. It is not failing the closer; it is invisible to it.\n\n"
        "**What is missing**, either of:\n\n"
        f"1. An OCC contract check — a `dod_evidence[].checks[]` entry in "
        f"`contracts/{ticket}.yaml`, landed by an evidence companion PR. This "
        "is the one the closer actually runs.\n"
        "2. A probe line in this description, on a line of its own, unbulleted:\n"
        "   `Probe: <command> => <observation that settles it>`\n"
        "   e.g. `Probe: uv run pytest tests/unit/test_x.py -q => exits 0`\n\n"
        "If the deliverable genuinely has no executable probe — a prose document, "
        "a ruling — then say so on the ticket: that is a decision to close it by "
        "hand, and it is worth recording as one rather than leaving it to look "
        "like an oversight.\n\n"
        "_No state was changed. This sweep only comments, and only once per "
        "ticket._"
    )


class LinearHygieneTransport:
    """Minimal Linear GraphQL client scoped to this sweep's needs.

    Carries the OMN-16106 retry policy for the reason measured there: an
    un-retried transient read drops the candidate from the run, and a
    fail-closed caller then reports that as a fact about the ticket.
    """

    # OMN-14951 gap 2: self-declared secret-ish env-var names read by this
    # boundary file (see scripts/check-env-reads.sh's check_secret_name_declarations).
    required_secrets: tuple[str, ...] = ("LINEAR_API_KEY",)

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 15.0,
        max_attempts: int = 4,
        base_delay_seconds: float = 1.0,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else os.environ.get("LINEAR_API_KEY", "")
        )
        self._timeout = timeout
        self._max_attempts = max(1, max_attempts)
        self._base_delay_seconds = max(0.0, base_delay_seconds)
        self.last_error: str = ""

    def apply_retry_policy(self, max_attempts: int, base_delay_seconds: float) -> None:
        """Adopt the contract-declared retry policy for this run."""
        self._max_attempts = max(1, max_attempts)
        self._base_delay_seconds = max(0.0, base_delay_seconds)

    def _backoff_seconds(self, attempt_index: int) -> float:
        window = min(
            self._base_delay_seconds * (2.0**attempt_index), _RETRY_MAX_DELAY_S
        )
        return window * (0.5 + random.random() / 2.0)

    async def query(
        self, query: str, variables: dict[str, object]
    ) -> dict[str, object] | None:
        """Run one GraphQL call. ``None`` on any unresolved failure."""
        if not self._api_key:
            self.last_error = "LINEAR_API_KEY is not set."
            return None
        headers = {"Authorization": self._api_key, "Content-Type": "application/json"}
        payload = {"query": query, "variables": variables}
        for attempt_index in range(self._max_attempts):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        _LINEAR_API_URL, json=payload, headers=headers
                    )
                status = response.status_code
                if status == _HTTP_TOO_MANY_REQUESTS or status >= _HTTP_SERVER_ERROR:
                    self.last_error = f"Linear API returned HTTP {status}."
                    retryable = True
                else:
                    response.raise_for_status()
                    data = response.json()
                    errors = data.get("errors")
                    if errors:
                        rendered = str(errors).lower()
                        self.last_error = f"Linear GraphQL errors: {str(errors)[:300]}"
                        retryable = any(
                            token in rendered for token in _RATE_LIMIT_TOKENS
                        )
                    else:
                        result = data.get("data")
                        if isinstance(result, dict):
                            self.last_error = ""
                            return result
                        self.last_error = "Linear API response carried no data object."
                        return None
            except httpx.TransportError as exc:
                self.last_error = sanitize_error_message(exc)
                retryable = True
            except (httpx.HTTPError, ValueError) as exc:
                self.last_error = sanitize_error_message(exc)
                return None
            if not retryable or attempt_index + 1 >= self._max_attempts:
                return None
            delay = self._backoff_seconds(attempt_index)
            if delay > 0.0:
                await asyncio.sleep(delay)
        return None

    async def fetch_in_progress(
        self, project: str, max_tickets: int
    ) -> list[dict[str, object]] | None:
        """Every In-Progress issue in scope, or ``None`` if unreadable."""
        issue_filter: dict[str, object] = {"state": {"type": {"eq": "started"}}}
        if project:
            issue_filter["project"] = {"id": {"eq": project}}
        collected: list[dict[str, object]] = []
        cursor: str | None = None
        while len(collected) < max_tickets:
            page = min(100, max_tickets - len(collected))
            data = await self.query(
                _IN_PROGRESS_QUERY,
                {"first": page, "after": cursor, "filter": issue_filter},
            )
            if data is None:
                return None
            issues = data.get("issues")
            if not isinstance(issues, dict):
                return None
            nodes = issues.get("nodes")
            if not isinstance(nodes, list):
                return None
            collected.extend(node for node in nodes if isinstance(node, dict))
            page_info = issues.get("pageInfo")
            page_info = page_info if isinstance(page_info, dict) else {}
            if not page_info.get("hasNextPage"):
                break
            end_cursor = page_info.get("endCursor")
            if not isinstance(end_cursor, str) or not end_cursor:
                break
            cursor = end_cursor
        return collected

    async def has_marker(self, issue_id: str, marker: str) -> bool | None:
        """Whether ``marker`` already appears in the issue's comments.

        ``None`` means "could not determine", NEVER "there are none"
        (OMN-16808): one releases a write and the other must block it.
        """
        cursor: str | None = None
        for _ in range(_MAX_COMMENT_PAGES):
            data = await self.query(
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
                if isinstance(node, dict) and marker in str(node.get("body") or ""):
                    return True
            page_info = connection.get("pageInfo")
            page_info = page_info if isinstance(page_info, dict) else {}
            if not page_info.get("hasNextPage"):
                return False
            end_cursor = page_info.get("endCursor")
            if not isinstance(end_cursor, str) or not end_cursor:
                return None
            cursor = end_cursor
        return None

    async def create_comment(self, issue_id: str, body: str) -> bool:
        data = await self.query(
            _COMMENT_CREATE_MUTATION, {"issueId": issue_id, "body": body}
        )
        if data is None:
            return False
        created = data.get("commentCreate")
        return bool(isinstance(created, dict) and created.get("success"))


class HandlerInProgressProbeHygiene:
    """Name every In-Progress ticket no mechanical path can close."""

    def __init__(self, linear: LinearHygieneTransport | None = None) -> None:
        self._linear = linear if linear is not None else LinearHygieneTransport()

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelProbeHygieneRequest
    ) -> ModelProbeHygieneResult:
        correlation_id = request.correlation_id or uuid4()
        if isinstance(self._linear, LinearHygieneTransport):
            self._linear.apply_retry_policy(
                request.linear_retry_max_attempts,
                request.linear_retry_base_delay_seconds,
            )

        excluded = {
            ticket.strip().upper()
            for ticket in request.exclude_tickets
            if ticket.strip()
        }

        issues = await self._linear.fetch_in_progress(
            request.project, request.max_tickets
        )
        if issues is None:
            return ModelProbeHygieneResult(
                correlation_id=correlation_id,
                dry_run=not request.apply,
                success=False,
                error_message=(
                    "Could not enumerate In-Progress issues from Linear"
                    + (
                        f" ({self._linear.last_error})"
                        if isinstance(self._linear, LinearHygieneTransport)
                        and self._linear.last_error
                        else ""
                    )
                    + ". Reporting nothing rather than reporting zero: an empty "
                    "sweep and a failed sweep have the same shape and opposite "
                    "meanings."
                ),
            )

        outcomes: list[ModelProbeHygieneOutcome] = []
        comments_left = request.max_comments_per_run
        for issue in issues:
            outcome, spent = await self._process_issue(
                issue=issue,
                request=request,
                excluded=excluded,
                comments_left=comments_left,
            )
            comments_left -= spent
            outcomes.append(outcome)

        return ModelProbeHygieneResult(
            correlation_id=correlation_id,
            dry_run=not request.apply,
            tickets_scanned=len(outcomes),
            tickets_with_probe=sum(
                1
                for outcome in outcomes
                if outcome.decision is EnumProbeHygieneDecision.HAS_PROBE
            ),
            tickets_without_probe=sum(
                1
                for outcome in outcomes
                if outcome.decision
                in (
                    EnumProbeHygieneDecision.COMMENTED,
                    EnumProbeHygieneDecision.SKIPPED_ALREADY_COMMENTED,
                    EnumProbeHygieneDecision.SKIPPED_DRY_RUN,
                    EnumProbeHygieneDecision.SKIPPED_COMMENT_BUDGET_EXHAUSTED,
                )
            ),
            tickets_commented=sum(1 for outcome in outcomes if outcome.comment_posted),
            tickets_skipped=sum(
                1
                for outcome in outcomes
                if outcome.decision.value.startswith("skipped_")
            ),
            tickets_errored=sum(
                1 for outcome in outcomes if outcome.decision.value.startswith("error_")
            ),
            outcomes=tuple(outcomes),
        )

    async def _process_issue(
        self,
        issue: dict[str, object],
        request: ModelProbeHygieneRequest,
        excluded: set[str],
        comments_left: int,
    ) -> tuple[ModelProbeHygieneOutcome, int]:
        """One ticket's verdict, and how much comment budget it spent."""
        ticket = str(issue.get("identifier") or "").strip()
        issue_id = str(issue.get("id") or "").strip()

        if ticket.upper() in excluded:
            return (
                ModelProbeHygieneOutcome(
                    ticket=ticket,
                    decision=EnumProbeHygieneDecision.SKIPPED_EXCLUDED,
                    reason=(
                        f"{ticket} is on the caller-supplied fence — another "
                        "lane holds it, so this sweep reached no verdict about it."
                    ),
                ),
                0,
            )

        description = str(issue.get("description") or "")
        probe_lines = _well_formed_probe_lines(description)
        contract_checks = _occ_contract_check_count(request.occ_repo_dir, ticket)

        if contract_checks is None:
            return (
                ModelProbeHygieneOutcome(
                    ticket=ticket,
                    decision=EnumProbeHygieneDecision.ERROR_CONTRACT_UNREADABLE,
                    reason=(
                        "The OCC contract could not be read, so 'declares no "
                        "check' could not be distinguished from 'could not "
                        "look'. Reported as an error, never as the finding."
                    ),
                    description_probe_lines=probe_lines,
                ),
                0,
            )

        if contract_checks > 0 or probe_lines > 0:
            return (
                ModelProbeHygieneOutcome(
                    ticket=ticket,
                    decision=EnumProbeHygieneDecision.HAS_PROBE,
                    reason=(
                        f"{contract_checks} OCC contract check(s) and "
                        f"{probe_lines} description probe line(s) — a "
                        "mechanical closing path exists."
                    ),
                    occ_contract_checks=contract_checks,
                    description_probe_lines=probe_lines,
                ),
                0,
            )

        # THE FINDING: no executable probe anywhere.
        seen = await self._linear.has_marker(issue_id, _HYGIENE_COMMENT_MARKER)
        if seen is None:
            return (
                ModelProbeHygieneOutcome(
                    ticket=ticket,
                    decision=EnumProbeHygieneDecision.ERROR_LINEAR_API,
                    reason=(
                        "Refusing to comment — the ticket's existing comments "
                        "could not be read, so the sweep cannot establish it "
                        "has not already said this here. Fails closed rather "
                        "than risk a duplicate (OMN-16808)."
                    ),
                ),
                0,
            )
        if seen:
            return (
                ModelProbeHygieneOutcome(
                    ticket=ticket,
                    decision=EnumProbeHygieneDecision.SKIPPED_ALREADY_COMMENTED,
                    reason=(
                        "Still has no executable probe; a previous run already "
                        "said so on this ticket. Reported so the standing list "
                        "stays visible, not re-commented."
                    ),
                ),
                0,
            )
        if not request.apply:
            return (
                ModelProbeHygieneOutcome(
                    ticket=ticket,
                    decision=EnumProbeHygieneDecision.SKIPPED_DRY_RUN,
                    reason="No executable probe. DRY RUN — no comment written.",
                ),
                0,
            )
        if comments_left <= 0:
            return (
                ModelProbeHygieneOutcome(
                    ticket=ticket,
                    decision=(
                        EnumProbeHygieneDecision.SKIPPED_COMMENT_BUDGET_EXHAUSTED
                    ),
                    reason=(
                        "No executable probe, and this run's comment budget is "
                        "spent. Not refused — its turn has not come round."
                    ),
                ),
                0,
            )
        posted = await self._linear.create_comment(issue_id, _comment_body(ticket))
        if not posted:
            return (
                ModelProbeHygieneOutcome(
                    ticket=ticket,
                    decision=EnumProbeHygieneDecision.ERROR_LINEAR_API,
                    reason="commentCreate did not report success.",
                ),
                0,
            )
        return (
            ModelProbeHygieneOutcome(
                ticket=ticket,
                decision=EnumProbeHygieneDecision.COMMENTED,
                reason=(
                    "No executable probe anywhere — commented once, naming what "
                    "is missing. No state changed."
                ),
                comment_posted=True,
            ),
            1,
        )


__all__ = ["HandlerInProgressProbeHygiene"]
