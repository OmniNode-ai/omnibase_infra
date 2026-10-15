# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The In Progress demotion leg of the evidence autoclose sweep (OMN-18489).

Why this exists
---------------
Work in progress was unbounded and more than half of it silent: a ticket was
entered into In Progress and abandoned, and nothing demoted, escalated or
counted it. The evidence closer already runs every two hours and already writes
Linear state; this is the decision branch beside its flip and gap branches that
returns an abandoned ticket to Backlog.

What it decides, and what it never does
---------------------------------------
A ticket is demoted only when ALL of these hold:

1. its state is In Progress;
2. no comment other than a sweep's own landed inside the window;
3. no open pull request cites it;
4. no live execution claim comment is on it;
5. it is not on the exclusion fence.

It also requires that the ticket entered In Progress before the window began,
because a ticket started an hour ago has not yet had the window to be touched.

It fails closed on every ambiguity. A Linear or GitHub read that fails, a
payload that cannot be interpreted, an open-pull-request search that returned
only part of its result: each leaves the ticket alone and is recorded as an
ERROR, never as a demotion. This leg moves In Progress to Backlog and nothing
else; it never touches a completed ticket, so the reopen-versus-successor
ruling is untouched. The comment it posts names the window and each resolution,
so the demotion is legible and reversible by anyone who disagrees: moving the
ticket back to In Progress resets ``startedAt`` and with it the window.

The window is a required field of the request. It has no default here; the
sweep's contract-declared ``demotion_window_hours`` supplies it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_in_progress_demotion_decision import (
    EnumInProgressDemotionDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_in_progress_demotion_outcome import (
    ModelInProgressDemotionOutcome,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_in_progress_demotion_request import (
    ModelInProgressDemotionRequest,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_in_progress_demotion_result import (
    ModelInProgressDemotionResult,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _LinearClient,
    )

logger = logging.getLogger(__name__)

_IN_PROGRESS_STATE_NAME = "In Progress"
_STARTED_STATE_TYPE = "started"
_BACKLOG_STATE_TYPE = "backlog"

#: The marker the demotion comment carries. It begins ``<!-- onex`` so that it,
#: like every other sweep comment, is discounted when the next run asks whether
#: anyone has touched the ticket.
DEMOTION_COMMENT_MARKER = "<!-- onex:in-progress-demotion -->"

# A sweep's own comment: ``<!-- onex-autoclose ...``, ``<!-- onex:in-progress-
# probe-hygiene ...`` and this leg's marker. Anything else is somebody's words.
_SWEEP_COMMENT_RE = re.compile(r"\A\s*<!--\s*onex[-:]")

# A live execution claim on the ticket: the lane-comment convention
# (``actor: <lane> (<model>)`` as the first line) or an explicit claim marker.
_EXECUTION_CLAIM_RE = re.compile(
    r"\A\s*(?:actor:\s*\S|<!--\s*onex:execution-claim\b)", re.IGNORECASE
)

_PR_SEARCH_PAGE_SIZE = 100
# GitHub's search API returns at most 1000 results for any query.
_PR_SEARCH_MAX_PAGES = 10

TypeRunGhCommand = Callable[[list[str], float], Awaitable[tuple[object | None, str]]]


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _ticket_cited_in(ticket_id: str, text: str) -> bool:
    """True when ``text`` names exactly ``ticket_id``, not a longer id or a suffix."""
    pattern = rf"(?<![A-Za-z0-9-]){re.escape(ticket_id)}(?![0-9])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _normalised_fence(request: ModelInProgressDemotionRequest) -> frozenset[str]:
    return frozenset(
        ticket.strip().upper() for ticket in request.exclude_tickets if ticket.strip()
    )


def _format_demotion_comment(ticket_id: str, window_hours: int) -> str:
    return (
        f"{DEMOTION_COMMENT_MARKER}\n"
        f"**{ticket_id} was moved from In Progress to Backlog by the scheduled "
        "evidence sweep.** Nothing on it had changed for "
        f"{window_hours} hours, and the sweep resolved each of these:\n\n"
        f"1. State: In Progress, entered more than {window_hours} hours ago.\n"
        f"2. Comment: none in the last {window_hours} hours other than a "
        "sweep's own.\n"
        "3. Pull request: no open pull request cites this ticket.\n"
        "4. Claim: no live execution claim comment is on it.\n"
        "5. Fence: it is not on the exclusion fence.\n\n"
        "If this is wrong, move it back to In Progress: the window starts "
        "again from that move. Only an unresolvable answer is ever read as "
        "permission to leave a ticket alone; no condition was guessed.\n"
    )


class InProgressDemotionLeg:
    """Move In Progress tickets that nothing is working on back to Backlog."""

    def __init__(
        self,
        linear_client: _LinearClient,
        run_gh_command: TypeRunGhCommand,
    ) -> None:
        self._linear = linear_client
        self._run_gh_command = run_gh_command

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelInProgressDemotionRequest
    ) -> ModelInProgressDemotionResult:
        now = datetime.now(tz=UTC)
        window = timedelta(hours=request.window_hours)
        since_iso = (now - window).strftime("%Y-%m-%dT%H:%M:%SZ")
        issues, enumeration_error = await self._linear.fetch_in_progress_issues(
            since_iso
        )
        if issues is None:
            return ModelInProgressDemotionResult(
                window_hours=request.window_hours,
                dry_run=not request.apply,
                success=False,
                error_message=(
                    "In Progress enumeration failed: "
                    f"{enumeration_error or 'no detail'}. No ticket was judged."
                ),
            )

        fence = _normalised_fence(request)
        open_prs = OpenPullRequestIndex(self._run_gh_command, request)
        backlog_ids: dict[str, str | None] = {}
        outcomes: list[ModelInProgressDemotionOutcome] = []
        writes_spent = 0

        for issue in issues:
            ticket_id = str(issue.get("identifier") or "").strip().upper()
            if not ticket_id:
                continue
            decision, reason = await self._judge(
                issue, ticket_id, now, window, request, fence, open_prs
            )
            if decision is None:
                # Every condition held.
                if writes_spent >= request.max_demotions_per_run:
                    decision, reason = (
                        EnumInProgressDemotionDecision.SKIPPED_BUDGET_EXHAUSTED,
                        f"Every condition held, but max_demotions_per_run="
                        f"{request.max_demotions_per_run} was already spent. "
                        "Offered again next run.",
                    )
                elif not request.apply:
                    decision, reason = (
                        EnumInProgressDemotionDecision.WOULD_DEMOTE,
                        "Every condition held; this run is a dry run, so "
                        "nothing was written.",
                    )
                else:
                    writes_spent += 1
                    decision, reason = await self._demote(
                        issue, ticket_id, request, backlog_ids
                    )
            logger.info("in_progress_demotion %s %s", decision.value, ticket_id)
            outcomes.append(
                ModelInProgressDemotionOutcome(
                    ticket_identifier=ticket_id, decision=decision, reason=reason
                )
            )

        def count(*members: EnumInProgressDemotionDecision) -> int:
            return sum(1 for outcome in outcomes if outcome.decision in members)

        return ModelInProgressDemotionResult(
            window_hours=request.window_hours,
            dry_run=not request.apply,
            candidates_scanned=len(outcomes),
            tickets_demoted=count(EnumInProgressDemotionDecision.DEMOTED),
            tickets_would_demote=count(EnumInProgressDemotionDecision.WOULD_DEMOTE),
            tickets_skipped=count(
                EnumInProgressDemotionDecision.SKIPPED_NOT_IN_PROGRESS,
                EnumInProgressDemotionDecision.SKIPPED_STARTED_WITHIN_WINDOW,
                EnumInProgressDemotionDecision.SKIPPED_RECENT_COMMENT,
                EnumInProgressDemotionDecision.SKIPPED_OPEN_PULL_REQUEST,
                EnumInProgressDemotionDecision.SKIPPED_LIVE_CLAIM,
                EnumInProgressDemotionDecision.SKIPPED_EXCLUDED,
                EnumInProgressDemotionDecision.SKIPPED_BUDGET_EXHAUSTED,
            ),
            tickets_errored=count(
                EnumInProgressDemotionDecision.ERROR_LINEAR_API,
                EnumInProgressDemotionDecision.ERROR_GITHUB_API,
                EnumInProgressDemotionDecision.ERROR_READBACK_UNCONFIRMED,
                EnumInProgressDemotionDecision.ERROR_COMMENT_UNPOSTED,
            ),
            outcomes=tuple(outcomes),
        )

    async def _judge(
        self,
        issue: dict[str, object],
        ticket_id: str,
        now: datetime,
        window: timedelta,
        request: ModelInProgressDemotionRequest,
        fence: frozenset[str],
        open_prs: OpenPullRequestIndex,
    ) -> tuple[EnumInProgressDemotionDecision, str] | tuple[None, str]:
        """The first condition that withholds the ticket, or ``(None, "")``.

        Ordered cheapest first: the fence and the state need no I/O, the
        comment evidence arrived with the enumeration, and the pull-request
        search runs only for a ticket that survived all of them.
        """
        decision = EnumInProgressDemotionDecision
        if ticket_id in fence:
            return decision.SKIPPED_EXCLUDED, (
                f"{ticket_id} is on the exclusion fence; refused before any other read."
            )
        state = issue.get("state")
        state = state if isinstance(state, dict) else {}
        if (
            state.get("type") != _STARTED_STATE_TYPE
            or state.get("name") != _IN_PROGRESS_STATE_NAME
        ):
            return decision.SKIPPED_NOT_IN_PROGRESS, (
                f"State is {state.get('name')!r} ({state.get('type')!r}), not "
                f"{_IN_PROGRESS_STATE_NAME!r}."
            )
        started_at = _parse_utc(issue.get("startedAt"))
        if started_at is None:
            return decision.ERROR_LINEAR_API, (
                "startedAt is missing or unreadable, so the time spent In "
                "Progress could not be resolved. Left alone."
            )
        if now - started_at < window:
            return decision.SKIPPED_STARTED_WITHIN_WINDOW, (
                f"Entered In Progress at {started_at.isoformat()}, inside the "
                f"{request.window_hours}-hour window."
            )

        comments = issue.get("comments")
        nodes = comments.get("nodes") if isinstance(comments, dict) else None
        page_info = comments.get("pageInfo") if isinstance(comments, dict) else None
        if not isinstance(nodes, list) or not isinstance(page_info, dict):
            return decision.ERROR_LINEAR_API, (
                "The comment connection was missing or malformed, so recent "
                "activity could not be resolved. Left alone."
            )
        bodies = [
            node.get("body") if isinstance(node, dict) else None for node in nodes
        ]
        if any(not isinstance(body, str) for body in bodies):
            return decision.ERROR_LINEAR_API, (
                "A comment in the window carried no readable body. Left alone."
            )
        texts = [body for body in bodies if isinstance(body, str)]
        if any(_EXECUTION_CLAIM_RE.search(text) for text in texts):
            return decision.SKIPPED_LIVE_CLAIM, (
                f"A live execution claim comment landed inside the "
                f"{request.window_hours}-hour window."
            )
        if page_info.get("hasNextPage") or any(
            not _SWEEP_COMMENT_RE.search(text) for text in texts
        ):
            return decision.SKIPPED_RECENT_COMMENT, (
                f"A comment other than a sweep's own landed inside the "
                f"{request.window_hours}-hour window."
            )

        cited, pr_error = await open_prs.cites(ticket_id)
        if pr_error:
            return decision.ERROR_GITHUB_API, (
                f"Open pull requests could not be resolved: {pr_error}. Left alone."
            )
        if cited is not None:
            return decision.SKIPPED_OPEN_PULL_REQUEST, (
                f"Open pull request #{cited} cites {ticket_id}."
            )
        return None, ""

    async def _demote(
        self,
        issue: dict[str, object],
        ticket_id: str,
        request: ModelInProgressDemotionRequest,
        backlog_ids: dict[str, str | None],
    ) -> tuple[EnumInProgressDemotionDecision, str]:
        decision = EnumInProgressDemotionDecision
        issue_id = str(issue.get("id") or "")
        team = issue.get("team")
        team_id = str(team.get("id") or "") if isinstance(team, dict) else ""
        if not issue_id or not team_id:
            return decision.ERROR_LINEAR_API, "The issue id or team id was missing."
        if team_id not in backlog_ids:
            backlog_ids[team_id] = await self._linear.fetch_backlog_state_id(team_id)
        backlog_id = backlog_ids[team_id]
        if not backlog_id:
            return decision.ERROR_LINEAR_API, (
                f"Team {team_id} has no Backlog state to move {ticket_id} to."
            )
        if not await self._linear.update_issue_state(issue_id, backlog_id):
            return decision.ERROR_LINEAR_API, "The Backlog state write failed."
        after = await self._linear.fetch_issue(ticket_id)
        after_state = after.get("state") if isinstance(after, dict) else None
        if (
            not isinstance(after_state, dict)
            or after_state.get("type") != _BACKLOG_STATE_TYPE
        ):
            return decision.ERROR_READBACK_UNCONFIRMED, (
                "The state write reported success but the readback did not "
                "show Backlog."
            )
        if not await self._linear.create_comment(
            issue_id, _format_demotion_comment(ticket_id, request.window_hours)
        ):
            return decision.ERROR_COMMENT_UNPOSTED, (
                "The ticket is in Backlog but the explanatory comment was not posted."
            )
        return decision.DEMOTED, (
            f"All five conditions held over {request.window_hours} hours; moved "
            "to Backlog and commented."
        )


class OpenPullRequestIndex:
    """Every open pull request of one organisation, read at most once per pass.

    One enumeration serves every candidate: a per-ticket search would spend the
    search API's per-minute budget on the first dozen tickets and fail closed
    on the rest. A failed or partial enumeration is cached as the error, so
    every candidate that needs it is left alone rather than judged on a
    fragment.
    """

    def __init__(
        self, run_gh_command: TypeRunGhCommand, request: ModelInProgressDemotionRequest
    ) -> None:
        self._run_gh_command = run_gh_command
        self._owner = request.pr_owner
        self._timeout = float(request.gh_timeout_seconds)
        self._loaded = False
        self._error = ""
        self._pulls: list[tuple[int, str]] = []

    async def cites(self, ticket_id: str) -> tuple[int | None, str]:
        """``(number, "")`` for a citing PR, ``(None, "")`` for none, ``(None, err)``."""
        if not self._loaded:
            await self._load()
        if self._error:
            return None, self._error
        for number, text in self._pulls:
            if _ticket_cited_in(ticket_id, text):
                return number, ""
        return None, ""

    async def _load(self) -> None:
        self._loaded = True
        pulls: list[tuple[int, str]] = []
        for page in range(1, _PR_SEARCH_MAX_PAGES + 1):
            path = (
                f"search/issues?q=org:{self._owner}+is:pr+is:open"
                f"&per_page={_PR_SEARCH_PAGE_SIZE}&page={page}"
            )
            data, error = await self._run_gh_command(["gh", "api", path], self._timeout)
            if data is None:
                self._error = error or "gh api returned no data"
                return
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list) or not isinstance(data, dict):
                self._error = "the search payload carried no 'items' list"
                return
            if data.get("incomplete_results"):
                self._error = "GitHub reported incomplete search results"
                return
            for item in items:
                if not isinstance(item, dict):
                    self._error = "the search payload carried a non-mapping item"
                    return
                number = item.get("number")
                pulls.append(
                    (
                        number if isinstance(number, int) else 0,
                        f"{item.get('title') or ''}\n{item.get('body') or ''}",
                    )
                )
            total = data.get("total_count")
            if not isinstance(total, int):
                self._error = "the search payload carried no total_count"
                return
            if len(pulls) >= total:
                self._pulls = pulls
                return
            if len(items) < _PR_SEARCH_PAGE_SIZE:
                break
        self._error = (
            f"the open-pull-request search returned {len(pulls)} of a larger "
            f"total within {_PR_SEARCH_MAX_PAGES} pages, so it cannot prove a "
            "ticket uncited"
        )


__all__ = [
    "DEMOTION_COMMENT_MARKER",
    "InProgressDemotionLeg",
]
