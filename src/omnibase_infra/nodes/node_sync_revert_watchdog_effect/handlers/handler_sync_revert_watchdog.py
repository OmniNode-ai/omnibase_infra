# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handler for the sync-revert watchdog sweep (OMN-16536).

Detects and corrects the Linear GitHub-sync silent Done-revert pattern
diagnosed in OMN-16536: a verified-Done ticket gets demoted to a
non-completed state by Linear's own GitHub-integration automation (a bare
`Refs:` mention in an unrelated PR, or a `ready_for_review` transition on
the ticket's own PR — both firing surfaces are documented in OMN-16536's
body and its F11/OMN-15751 comment), with no human ever making the call.

Detection signal
-----------------
Linear's public GraphQL schema documents the automation signature
directly on ``IssueHistory``:

    actorId: "... Null if the change was made by an integration,
    automation, or system process."
    botActor: "The bot that performed the action."
    botActor.type: "... identifying the application or integration
    (e.g., 'github', 'slack', 'workflow', 'ai')."

So a history entry with ``actorId`` null and ``fromState.type ==
"completed"`` / ``toState.type != "completed"`` is exactly the shape of a
silent automation-driven Done-revert. That is the PRIMARY signal here —
not a timing heuristic — corroborated by two additional guards so the
watchdog never overrides a genuine human decision:

1. No human-authored comment (excluding the watchdog's own prior
   diagnosis comments) inside ``human_comment_window_seconds`` of the
   transition — the literal "no human comment between" condition this
   node was scoped to implement.
2. No LATER state-change history entry with a non-null ``actorId`` — if a
   human has since looked at the ticket and made a further workflow-state
   change, the watchdog defers to that person rather than re-flipping
   over their decision. (Scoped to state-change entries specifically — a
   human editing an unrelated field such as a label or assignee after the
   automation revert must not suppress the correction.)
3. The completed state the re-flip would RESTORE must itself have been
   set by a human (OMN-16762). Restoring a Done that automation set
   reinstates an automation artifact rather than a human decision —
   exactly the outcome this watchdog exists to prevent. Automation-set
   and indeterminate provenance both fail closed as
   ``SKIPPED_PRIOR_DONE_NOT_HUMAN_SET``; see ``EnumPriorDoneActorKind``.

History window (OMN-16762)
--------------------------
Per-ticket history is walked FORWARD (``first``/``after``) to the true
end of the connection, never with ``last``. Linear's
``orderBy: createdAt`` sorts DESCENDING, so ``last: N`` returns the N
OLDEST entries — the opposite of what this node's request model used to
claim. That defect made the sweep classify a stale revert on any ticket
with more history than one page, and left guards 1 and 2 above scanning
a window that could not contain the evidence they look for: across 126
detected reverts in the OMN-16536 adjudication they fired zero times.

Pipeline
--------
1. Kill switch: if ``ONEX_SYNC_REVERT_WATCHDOG_DISABLED`` is set, do zero
   I/O and return.
2. Enumerate the team's tickets updated within ``lookback_hours`` (a
   state revert bumps ``updatedAt``, so this is a correct, cheap filter —
   no need to walk every ticket on the team every run).
3. For each ticket, walk its full history newest-first and find the
   latest completed->non-completed transition, if any.
4. Classify per the signals above. ``ERROR_STATE_NOT_RESOLVABLE`` fires
   when the prior completed state no longer resolves live on the team
   (renamed/deleted workflow state) — never guessed, never substituted.
5. ``apply=False`` (the default) performs every read above but never
   calls a Linear mutation — every decision is logged as "would-do".
   ``apply=True`` performs the real ``issueUpdate``/``commentCreate``.

Non-blocking Design
--------------------
Per-ticket failures (Linear API errors) are recorded in the outcome list
and do not abort the sweep — only a sweep-level failure (issue
enumeration itself failing) sets ``result.success = False``.

This node is a complementary safety net (fix option 3 in OMN-16536's
body), not a substitute for fixing the triggering ``GitAutomationState``
config directly (fix options 1/2, still open, separate work).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.enum_prior_done_actor_kind import (
    EnumPriorDoneActorKind,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.enum_sync_revert_watchdog_decision import (
    EnumSyncRevertWatchdogDecision,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.model_sync_revert_watchdog_outcome import (
    ModelSyncRevertWatchdogOutcome,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.model_sync_revert_watchdog_request import (
    ModelSyncRevertWatchdogRequest,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.model_sync_revert_watchdog_result import (
    ModelSyncRevertWatchdogResult,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

__all__ = ["HandlerSyncRevertWatchdog"]

# Injectable Linear GraphQL query type (real impl calls httpx; tests inject fakes).
TypeLinearQuery = Callable[
    [str, dict[str, object], float], Awaitable[dict[str, object] | None]
]

# Kill switch env var (checked first, unconditionally — mirrors OMN-16106's
# ONEX_AUTOCLOSE_DISABLED precedent, deliberately a SEPARATE variable so
# disabling one sweep never silently disables the other).
_KILL_SWITCH_ENV_VAR = "ONEX_SYNC_REVERT_WATCHDOG_DISABLED"

_LINEAR_API_URL = "https://api.linear.app/graphql"  # url-authority-ok: fixed public GraphQL API, no ONEX routing authority

_COMPLETED_TYPE = "completed"

_RECENT_ISSUES_QUERY = """
query RecentIssues($teamKey: String!, $since: DateTimeOrDuration!, $first: Int!, $after: String) {
  issues(
    filter: { team: { key: { eq: $teamKey } }, updatedAt: { gte: $since } }
    first: $first
    after: $after
    orderBy: updatedAt
  ) {
    nodes {
      id
      identifier
      state { id name type }
      team { id }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

# OMN-16762: FORWARD pagination (`first`/`after`), never `last`.
#
# Linear's `orderBy: createdAt` sorts the connection DESCENDING, so
# `first: N` yields the N NEWEST entries and `last: N` yields the N
# OLDEST. This query previously used `last`, which meant the sweep read
# the oldest page of every ticket's history and could not see anything
# newer. Measured live against OMN-14888 (553 entries) on 2026-08-27:
#
#   history(last: 50,  orderBy: createdAt) -> 2026-07-21 .. 2026-07-26
#   history(first: 50, orderBy: createdAt) -> 2026-08-27 .. 2026-08-25
#
# Walking `first`/`after` to pageInfo exhaustion is the robust option the
# probe confirmed: 553 entries came back over 3 pages of 250, in
# newest-first order. Full history is what the classifier needs anyway —
# the pre-revert-Done provenance check reads BACKWARD from the revert.
_ISSUE_HISTORY_QUERY = """
query IssueHistory($issueId: String!, $first: Int!, $after: String) {
  issue(id: $issueId) {
    history(first: $first, after: $after, orderBy: createdAt) {
      nodes {
        id
        createdAt
        actorId
        botActor { type name }
        fromState { id name type }
        toState { id name type }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

_ISSUE_COMMENTS_QUERY = """
query IssueComments($issueId: String!) {
  issue(id: $issueId) {
    comments(first: 250) {
      nodes {
        id
        createdAt
        body
        user { id }
      }
    }
  }
}
"""

_TEAM_STATES_QUERY = """
query TeamStates($teamId: String!) {
  team(id: $teamId) {
    id
    states(first: 100) { nodes { id name type } }
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


def _parse_iso(value: object) -> datetime | None:
    """Best-effort ISO-8601 -> aware datetime. None on any parse failure."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class _LinearClient:
    """Minimal Linear GraphQL client scoped to this watchdog's needs.

    Reads ``LINEAR_API_KEY`` from the environment when not passed
    explicitly (mirrors ``node_evidence_autoclose_sweep_effect``'s
    ``_LinearClient``, itself mirroring ``GitHubTransport.__init__``'s
    ``GH_PAT`` fallback in ``omnibase_infra.adapters.github``).
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
        self, query: str, variables: dict[str, object], timeout: float
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
            async with httpx.AsyncClient(timeout=timeout or self._timeout) as client:
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

    async def fetch_recently_updated_issues(
        self, team_key: str, since_iso: str, max_issues: int, timeout: float
    ) -> tuple[list[dict[str, object]], str]:
        """Paginated enumeration of tickets updated since ``since_iso``."""
        issues: list[dict[str, object]] = []
        cursor: str | None = None
        max_pages = 40  # safety cap against unbounded pagination
        for _ in range(max_pages):
            variables: dict[str, object] = {
                "teamKey": team_key,
                "since": since_iso,
                "first": min(250, max_issues - len(issues)),
                "after": cursor,
            }
            data = await self._query(_RECENT_ISSUES_QUERY, variables, timeout)
            if data is None:
                return issues, "Failed to fetch recently-updated issues from Linear."
            connection = data.get("issues")
            nodes = connection.get("nodes") if isinstance(connection, dict) else None
            if not isinstance(nodes, list):
                return issues, "Malformed issues connection in Linear response."
            issues.extend(n for n in nodes if isinstance(n, dict))
            if len(issues) >= max_issues:
                return issues[:max_issues], ""
            page_info = (
                connection.get("pageInfo") if isinstance(connection, dict) else None
            )
            has_next = bool(
                isinstance(page_info, dict) and page_info.get("hasNextPage")
            )
            if not has_next:
                break
            cursor = (
                str(page_info.get("endCursor")) if isinstance(page_info, dict) else None
            )
            if not cursor:
                break
        return issues, ""

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int, timeout: float
    ) -> tuple[list[dict[str, object]] | None, str]:
        """One ticket's history, walked newest-first to the true end.

        Paginates forward via ``first``/``after`` until Linear reports no
        further page or ``max_pages`` is reached. Because the walk runs
        newest-first, hitting the cap truncates the OLDEST end only — the
        entries every guard depends on (the latest revert, any later
        human state change) are always present. A pre-revert Done that
        falls outside a capped walk resolves to
        ``EnumPriorDoneActorKind.UNKNOWN``, which fails closed.

        Returns ``(None, error)`` on any failed page — a partial history
        must never be classified as if it were complete.
        """
        collected: list[dict[str, object]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            variables: dict[str, object] = {
                "issueId": issue_id,
                "first": page_size,
                "after": cursor,
            }
            data = await self._query(_ISSUE_HISTORY_QUERY, variables, timeout)
            if data is None:
                return None, "Failed to fetch issue history from Linear."
            issue = data.get("issue")
            history = issue.get("history") if isinstance(issue, dict) else None
            nodes = history.get("nodes") if isinstance(history, dict) else None
            if not isinstance(nodes, list):
                return None, "Malformed history connection in Linear response."
            collected.extend(n for n in nodes if isinstance(n, dict))
            page_info = history.get("pageInfo") if isinstance(history, dict) else None
            if not (isinstance(page_info, dict) and page_info.get("hasNextPage")):
                break
            next_cursor = page_info.get("endCursor")
            if not next_cursor:
                break
            cursor = str(next_cursor)
        else:
            logger.warning(
                "Issue %s history walk hit the %d-page cap — oldest entries "
                "truncated; pre-revert Done provenance may fail closed as UNKNOWN.",
                issue_id,
                max_pages,
            )
        return collected, ""

    async def fetch_issue_comments(
        self, issue_id: str, timeout: float
    ) -> tuple[list[dict[str, object]] | None, str]:
        """All comments (bounded to 250) for one ticket. None on failure."""
        data = await self._query(_ISSUE_COMMENTS_QUERY, {"issueId": issue_id}, timeout)
        if data is None:
            return None, "Failed to fetch issue comments from Linear."
        issue = data.get("issue")
        comments = issue.get("comments") if isinstance(issue, dict) else None
        nodes = comments.get("nodes") if isinstance(comments, dict) else None
        if not isinstance(nodes, list):
            return None, "Malformed comments connection in Linear response."
        return [n for n in nodes if isinstance(n, dict)], ""

    async def fetch_team_states(
        self, team_id: str, timeout: float
    ) -> list[dict[str, object]] | None:
        """All workflow states for a team. None on any failure."""
        data = await self._query(_TEAM_STATES_QUERY, {"teamId": team_id}, timeout)
        if data is None:
            return None
        team = data.get("team")
        states_conn = team.get("states") if isinstance(team, dict) else None
        nodes = states_conn.get("nodes") if isinstance(states_conn, dict) else None
        return (
            [n for n in nodes if isinstance(n, dict)]
            if isinstance(nodes, list)
            else None
        )

    async def update_issue_state(
        self, issue_id: str, state_id: str, timeout: float
    ) -> bool:
        """Flip an issue to the given workflow state. False on any failure."""
        data = await self._query(
            _ISSUE_UPDATE_STATE_MUTATION,
            {"issueId": issue_id, "stateId": state_id},
            timeout,
        )
        if data is None:
            return False
        update = data.get("issueUpdate")
        return bool(isinstance(update, dict) and update.get("success"))

    async def create_comment(self, issue_id: str, body: str, timeout: float) -> bool:
        """Post a comment on an issue. False on any failure."""
        data = await self._query(
            _COMMENT_CREATE_MUTATION, {"issueId": issue_id, "body": body}, timeout
        )
        if data is None:
            return False
        created = data.get("commentCreate")
        return bool(isinstance(created, dict) and created.get("success"))


class HandlerSyncRevertWatchdog:
    """Sweep a team's recently-updated tickets and correct silent Done-reverts."""

    def __init__(
        self,
        linear_client: _LinearClient | None = None,
        kill_switch_disabled: bool | None = None,
    ) -> None:
        # ``kill_switch_disabled`` mirrors the evidence-autoclose-sweep
        # precedent: read at construction time, override injectable for
        # tests. Re-checked defensively at the top of handle() too so a
        # zero-arg contract-driven construction can never silently skip it.
        self._linear = linear_client if linear_client is not None else _LinearClient()
        self._kill_switch_ctor = (
            kill_switch_disabled
            if kill_switch_disabled is not None
            else bool(os.environ.get(_KILL_SWITCH_ENV_VAR, ""))
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    # -- main entrypoint ---------------------------------------------------

    async def handle(
        self, request: ModelSyncRevertWatchdogRequest
    ) -> ModelSyncRevertWatchdogResult:
        correlation_id = request.correlation_id or uuid4()

        kill_switch_engaged = self._kill_switch_ctor or bool(
            os.environ.get(_KILL_SWITCH_ENV_VAR, "")
        )
        if kill_switch_engaged:
            logger.warning(
                "%s is set — sync-revert watchdog disabled, zero I/O performed.",
                _KILL_SWITCH_ENV_VAR,
            )
            return ModelSyncRevertWatchdogResult(
                correlation_id=correlation_id,
                dry_run=not request.apply,
                kill_switch_engaged=True,
            )

        since_iso = (
            datetime.now(tz=UTC) - timedelta(hours=request.lookback_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        issues, enum_error = await self._linear.fetch_recently_updated_issues(
            request.team_key,
            since_iso,
            request.max_issues,
            request.linear_timeout_seconds,
        )
        if enum_error:
            return ModelSyncRevertWatchdogResult(
                correlation_id=correlation_id,
                dry_run=not request.apply,
                success=False,
                error_message=f"Issue enumeration failed: {enum_error}",
            )

        outcomes: list[ModelSyncRevertWatchdogOutcome] = []
        for issue in issues:
            outcomes.append(await self._process_issue(issue, request))

        # Any outcome that reached a revert transition at all (regardless
        # of whether it was ultimately corrected) carries a non-empty
        # `reverted_at` — computed from the field, not a hand-maintained
        # enum subset, so a future decision value can never silently drop
        # out of this count.
        reverts_detected = sum(1 for o in outcomes if o.reverted_at)
        reflipped = sum(
            1
            for o in outcomes
            if o.decision == EnumSyncRevertWatchdogDecision.REFLIPPED
        )
        skipped = sum(
            1
            for o in outcomes
            if o.decision
            in (
                EnumSyncRevertWatchdogDecision.SKIPPED_NO_REVERT_FOUND,
                EnumSyncRevertWatchdogDecision.SKIPPED_HUMAN_ACTOR,
                EnumSyncRevertWatchdogDecision.SKIPPED_HUMAN_COMMENT_NEARBY,
                EnumSyncRevertWatchdogDecision.SKIPPED_STATE_CHANGED_SINCE,
                EnumSyncRevertWatchdogDecision.SKIPPED_ALREADY_RESOLVED,
                EnumSyncRevertWatchdogDecision.SKIPPED_PRIOR_DONE_NOT_HUMAN_SET,
            )
        )
        errored = sum(
            1
            for o in outcomes
            if o.decision
            in (
                EnumSyncRevertWatchdogDecision.ERROR_LINEAR_API,
                EnumSyncRevertWatchdogDecision.ERROR_STATE_NOT_RESOLVABLE,
            )
        )

        return ModelSyncRevertWatchdogResult(
            correlation_id=correlation_id,
            dry_run=not request.apply,
            issues_scanned=len(issues),
            reverts_detected=reverts_detected,
            tickets_reflipped=reflipped,
            tickets_skipped=skipped,
            tickets_errored=errored,
            outcomes=tuple(outcomes),
        )

    async def _process_issue(
        self, issue: dict[str, object], request: ModelSyncRevertWatchdogRequest
    ) -> ModelSyncRevertWatchdogOutcome:
        ticket_id = str(issue.get("identifier") or "")
        issue_id = str(issue.get("id") or "")
        current_state = issue.get("state")
        current_state_type = (
            str(current_state.get("type", ""))
            if isinstance(current_state, dict)
            else ""
        )

        history, history_error = await self._linear.fetch_issue_history(
            issue_id,
            request.history_page_size,
            request.history_max_pages,
            request.linear_timeout_seconds,
        )
        if history is None:
            return ModelSyncRevertWatchdogOutcome(
                ticket_id=ticket_id,
                decision=EnumSyncRevertWatchdogDecision.ERROR_LINEAR_API,
                reason=history_error,
            )

        # Never trust the API's return order — re-sort by createdAt.
        dated_history = [
            (parsed, entry)
            for entry in history
            if (parsed := _parse_iso(entry.get("createdAt"))) is not None
        ]
        dated_history.sort(key=lambda pair: pair[0])

        revert_entries: list[
            tuple[datetime, dict[str, object], dict[str, object], dict[str, object]]
        ] = []
        for created_at, entry in dated_history:
            from_state_candidate = entry.get("fromState")
            to_state_candidate = entry.get("toState")
            if not isinstance(from_state_candidate, dict) or not isinstance(
                to_state_candidate, dict
            ):
                continue
            if (
                from_state_candidate.get("type") != _COMPLETED_TYPE
                or to_state_candidate.get("type") == _COMPLETED_TYPE
            ):
                continue
            revert_entries.append(
                (created_at, entry, from_state_candidate, to_state_candidate)
            )
        if not revert_entries:
            return ModelSyncRevertWatchdogOutcome(
                ticket_id=ticket_id,
                decision=EnumSyncRevertWatchdogDecision.SKIPPED_NO_REVERT_FOUND,
                reason="No completed->non-completed transition in scanned history.",
            )

        revert_at, revert_entry, from_state, to_state = revert_entries[-1]
        bot_actor = revert_entry.get("botActor")
        bot_actor_type = (
            str(bot_actor.get("type", "")) if isinstance(bot_actor, dict) else ""
        )

        # OMN-16762: establish WHO set the completed state this revert
        # would restore. Read backward from the revert for the most recent
        # transition INTO a completed state; its actorId decides. Recorded
        # on every revert-bearing outcome so an armed run's candidate set
        # can be audited mechanically, not just the ones it gates.
        def _sets_completed_state(entry: dict[str, object]) -> bool:
            to_state_candidate = entry.get("toState")
            return (
                isinstance(to_state_candidate, dict)
                and to_state_candidate.get("type") == _COMPLETED_TYPE
            )

        prior_done_entry = next(
            (
                entry
                for created_at, entry in reversed(dated_history)
                if created_at < revert_at and _sets_completed_state(entry)
            ),
            None,
        )
        if prior_done_entry is None:
            prior_done_kind = EnumPriorDoneActorKind.UNKNOWN
        elif prior_done_entry.get("actorId"):
            prior_done_kind = EnumPriorDoneActorKind.HUMAN
        else:
            prior_done_kind = EnumPriorDoneActorKind.BOT

        common_fields: dict[str, object] = {
            "ticket_id": ticket_id,
            "reverted_at": revert_entry.get("createdAt") or "",
            "from_state_name": str(from_state.get("name", "")),
            "to_state_name": str(to_state.get("name", "")),
            "bot_actor_type": bot_actor_type,
            "prior_done_actor_kind": prior_done_kind,
            "prior_done_at": (
                str(prior_done_entry.get("createdAt") or "")
                if prior_done_entry is not None
                else ""
            ),
        }

        if current_state_type == _COMPLETED_TYPE:
            return ModelSyncRevertWatchdogOutcome(
                decision=EnumSyncRevertWatchdogDecision.SKIPPED_ALREADY_RESOLVED,
                reason="Ticket's current state is already completed-type.",
                **common_fields,
            )

        if revert_entry.get("actorId"):
            return ModelSyncRevertWatchdogOutcome(
                decision=EnumSyncRevertWatchdogDecision.SKIPPED_HUMAN_ACTOR,
                reason="Revert transition's actorId was set — a real Linear user made this change.",
                **common_fields,
            )

        # Scoped to STATE-CHANGE entries only (toState present) — a human
        # editing an unrelated field (label, assignee, priority) after the
        # automation revert must not suppress the correction; only a
        # later human-driven WORKFLOW STATE change counts as "a person has
        # since made a deliberate call about this ticket's status."
        later_human_state_changes = [
            entry
            for created_at, entry in dated_history
            if created_at > revert_at
            and entry.get("actorId")
            and entry.get("toState") is not None
        ]
        if later_human_state_changes:
            return ModelSyncRevertWatchdogOutcome(
                decision=EnumSyncRevertWatchdogDecision.SKIPPED_STATE_CHANGED_SINCE,
                reason="A human made a further state change after the detected revert — deferring to that decision.",
                **common_fields,
            )

        # OMN-16762 restore precondition. Checked BEFORE the comments
        # round trip because it is pure history — no extra Linear call.
        # Only a human-set prior Done may be restored; automation-set and
        # indeterminate both fail closed. Deliberately ordered AFTER the
        # later-human-state-change guard so a ticket a person has since
        # ruled on is still attributed to that human decision rather than
        # to this precondition.
        if prior_done_kind is not EnumPriorDoneActorKind.HUMAN:
            return ModelSyncRevertWatchdogOutcome(
                decision=EnumSyncRevertWatchdogDecision.SKIPPED_PRIOR_DONE_NOT_HUMAN_SET,
                reason=(
                    "The completed state this revert would restore was "
                    f"{'set by automation' if prior_done_kind is EnumPriorDoneActorKind.BOT else 'of indeterminate provenance'}"
                    f" (prior_done_actor_kind={prior_done_kind.value}) — "
                    "restoring it would reinstate an automation artifact, "
                    "not a human decision. Only a human-set or formally "
                    "adjudicated Done is restorable."
                ),
                **common_fields,
            )

        comments, comments_error = await self._linear.fetch_issue_comments(
            issue_id, request.linear_timeout_seconds
        )
        if comments is None:
            return ModelSyncRevertWatchdogOutcome(
                decision=EnumSyncRevertWatchdogDecision.ERROR_LINEAR_API,
                reason=comments_error,
                **common_fields,
            )
        window = timedelta(seconds=request.human_comment_window_seconds)
        human_comment_nearby = False
        for comment in comments:
            comment_at = _parse_iso(comment.get("createdAt"))
            if comment_at is None:
                continue
            user = comment.get("user")
            is_human = isinstance(user, dict) and bool(user.get("id"))
            is_watchdog_comment = request.watchdog_comment_marker in str(
                comment.get("body") or ""
            )
            if (
                is_human
                and not is_watchdog_comment
                and abs((comment_at - revert_at).total_seconds())
                <= window.total_seconds()
            ):
                human_comment_nearby = True
                break
        if human_comment_nearby:
            return ModelSyncRevertWatchdogOutcome(
                decision=EnumSyncRevertWatchdogDecision.SKIPPED_HUMAN_COMMENT_NEARBY,
                reason="A human comment falls inside the detection window around the revert — treated as explained.",
                **common_fields,
            )

        # Confirmed silent automation revert. Resolve the target (prior
        # completed) state — never guessed, must still resolve live.
        team = issue.get("team")
        team_id = str(team.get("id") or "") if isinstance(team, dict) else ""
        target_state_id = str(from_state.get("id") or "")
        reason = (
            f"Automation-driven revert detected: {from_state.get('name')} -> "
            f"{to_state.get('name')} at {revert_entry.get('createdAt')}, "
            f"actorId=null, botActor.type={bot_actor_type or '(none)'}, "
            "no human comment nearby, no later human state change."
        )

        if not request.apply:
            logger.info(
                "[DRY-RUN] Would re-flip %s to %s (%s)",
                ticket_id,
                from_state.get("name"),
                reason,
            )
            return ModelSyncRevertWatchdogOutcome(
                decision=EnumSyncRevertWatchdogDecision.REFLIPPED,
                reason=f"[DRY-RUN] {reason}",
                applied=False,
                **common_fields,
            )

        team_states = await self._linear.fetch_team_states(
            team_id, request.linear_timeout_seconds
        )
        target_still_valid = bool(
            team_states
            and any(
                s.get("id") == target_state_id and s.get("type") == _COMPLETED_TYPE
                for s in team_states
            )
        )
        if not target_still_valid:
            return ModelSyncRevertWatchdogOutcome(
                decision=EnumSyncRevertWatchdogDecision.ERROR_STATE_NOT_RESOLVABLE,
                reason=(
                    f"Prior completed state '{from_state.get('name')}' "
                    f"({target_state_id}) no longer resolves live on the team."
                ),
                **common_fields,
            )

        flipped = await self._linear.update_issue_state(
            issue_id, target_state_id, request.linear_timeout_seconds
        )
        if not flipped:
            return ModelSyncRevertWatchdogOutcome(
                decision=EnumSyncRevertWatchdogDecision.ERROR_LINEAR_API,
                reason="issueUpdate(stateId) mutation failed.",
                **common_fields,
            )

        comment_body = (
            f"Automatic re-flip — {request.watchdog_comment_marker}.\n\n"
            f"Detected a silent automation-driven revert: **{from_state.get('name')}** "
            f"-> **{to_state.get('name')}** at `{revert_entry.get('createdAt')}` "
            f"(actorId null, botActor.type=`{bot_actor_type or '(none)'}` — Linear's own "
            "integration/automation signature). No human comment fell inside the "
            f"{request.human_comment_window_seconds}s detection window and no later "
            "human-driven state change was found, so this is treated as unexplained "
            f"and re-flipped back to **{from_state.get('name')}**.\n\n"
            "This is a complementary correction, not a fix for the triggering "
            "GitAutomationState config — see OMN-16536 fix options 1/2."
        )
        commented = await self._linear.create_comment(
            issue_id, comment_body, request.linear_timeout_seconds
        )
        return ModelSyncRevertWatchdogOutcome(
            decision=EnumSyncRevertWatchdogDecision.REFLIPPED,
            reason=reason,
            linear_comment_posted=commented,
            applied=True,
            **common_fields,
        )
