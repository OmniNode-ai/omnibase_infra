# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16762 regression suite — the history window must reach the TRUE newest entry.

Background (OMN-16536 revert-backlog adjudication,
``docs/tracking/2026-08-27-revert-backlog-adjudication.md``): the watchdog
fetched per-ticket history with ``history(last: $n, orderBy: createdAt)``
and the request model documented that as "the N MOST RECENT entries". It
is the opposite. Linear's ``orderBy: createdAt`` sorts DESCENDING, so
``last: N`` returns the tail of the descending list — the N **OLDEST**
entries.

Measured live against OMN-14888 (553 history entries) on 2026-08-27:

* ``history(last: 50, orderBy: createdAt)`` returned
  2026-07-21T02:46:36Z .. 2026-07-26T08:59:48Z — the 50 OLDEST.
* ``history(first: 50, orderBy: createdAt)`` returned
  2026-08-27T13:38:37Z .. 2026-08-25T03:44:16Z — the 50 NEWEST.
* ``first``/``after`` walked to exhaustion returned all 553 entries over
  3 pages of 250, newest-first.

Consequences proven by that audit: on 26 of 126 audited tickets history
exceeds 50 entries, and on 9 of them the watchdog named a STALE revert
rather than the true most-recent one. Both "no later human state change"
and "no human comment nearby" guards scanned the same truncated window
and fired **0 times across 126 detected reverts**.

``FakeLinearTransport`` below reproduces those measured semantics exactly,
so these tests fail against the truncating query and pass only against a
forward-paginating one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_sync_revert_watchdog_effect.handlers.handler_sync_revert_watchdog import (
    HandlerSyncRevertWatchdog,
    _LinearClient,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.enum_prior_done_actor_kind import (
    EnumPriorDoneActorKind,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.enum_sync_revert_watchdog_decision import (
    EnumSyncRevertWatchdogDecision,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.model_sync_revert_watchdog_request import (
    ModelSyncRevertWatchdogRequest,
)

_TEAM_ID = "team-omn-uuid"
_DONE_STATE_ID = "state-done-uuid"
_STARTED_STATE_ID = "state-started-uuid"
_REVIEW_STATE_ID = "state-review-uuid"
_HUMAN_ACTOR_ID = "7a850ce1-f95e-431f-b4e3-62f7449f04c0"  # Jonah Gray, per the audit

# The real OMN-14888 timestamps the audit recorded, used verbatim so this
# fixture stays tied to the incident it was derived from.
_STALE_REVERT_AT = datetime(2026, 7, 26, 8, 57, 13, tzinfo=UTC)
_TRUE_PRIOR_DONE_AT = datetime(2026, 7, 29, 2, 21, 42, 190000, tzinfo=UTC)
_TRUE_REVERT_AT = datetime(2026, 7, 29, 4, 33, 59, 627000, tzinfo=UTC)
_LATER_HUMAN_CHANGE_AT = datetime(2026, 8, 27, 13, 38, 37, 577000, tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _entry(
    *,
    created_at: datetime,
    from_state: tuple[str, str, str] | None = None,
    to_state: tuple[str, str, str] | None = None,
    actor_id: str | None = None,
    bot_type: str = "",
) -> dict[str, object]:
    """One IssueHistory node. ``*_state`` tuples are (id, name, type)."""

    def _state(spec: tuple[str, str, str] | None) -> dict[str, object] | None:
        if spec is None:
            return None
        return {"id": spec[0], "name": spec[1], "type": spec[2]}

    return {
        "id": str(uuid4()),
        "createdAt": _iso(created_at),
        "actorId": actor_id,
        "botActor": {"type": bot_type, "name": "GitHub"} if bot_type else None,
        "fromState": _state(from_state),
        "toState": _state(to_state),
    }


_DONE = (_DONE_STATE_ID, "Done", "completed")
_PROG = (_STARTED_STATE_ID, "In Progress", "started")
_REVIEW = (_REVIEW_STATE_ID, "In Review", "started")


def _bot_revert(created_at: datetime) -> dict[str, object]:
    return _entry(
        created_at=created_at,
        from_state=_DONE,
        to_state=_PROG,
        actor_id=None,
        bot_type="integration",
    )


def _bot_done(created_at: datetime) -> dict[str, object]:
    return _entry(
        created_at=created_at,
        from_state=_PROG,
        to_state=_DONE,
        actor_id=None,
        bot_type="integration",
    )


def _human_done(created_at: datetime) -> dict[str, object]:
    return _entry(
        created_at=created_at,
        from_state=_PROG,
        to_state=_DONE,
        actor_id=_HUMAN_ACTOR_ID,
    )


def _filler(created_at: datetime) -> dict[str, object]:
    """A bot flap entry that is NOT a completed->non-completed transition."""
    return _entry(
        created_at=created_at,
        from_state=_PROG,
        to_state=_REVIEW,
        actor_id=None,
        bot_type="integration",
    )


def _omn14888_shaped_history(
    *, include_later_human_change: bool
) -> list[dict[str, object]]:
    """A 553-entry history modelled on OMN-14888, ascending by createdAt.

    Layout is deliberate — the first 50 entries (which is exactly what the
    defective ``last: 50`` query returns) contain a STALE revert whose
    pre-revert Done was HUMAN-set, so the truncating implementation
    reaches a confident "re-flip this" verdict. The TRUE most-recent
    revert sits outside that window and has a BOT-set pre-revert Done,
    plus (optionally) a later human state change after it.
    """
    entries: list[dict[str, object]] = []

    # -- oldest 50: the window the defective query sees ------------------
    base = datetime(2026, 7, 21, 2, 46, 36, 282000, tzinfo=UTC)
    for i in range(48):
        entries.append(_filler(base + timedelta(hours=i * 2)))
    # A human-set Done, then a bot revert of it: inside the truncated
    # window this looks like a clean, restorable correction.
    entries.append(_human_done(_STALE_REVERT_AT - timedelta(minutes=7)))
    entries.append(_bot_revert(_STALE_REVERT_AT))
    assert len(entries) == 50

    # -- the TRUE revert cluster, outside the truncated window -----------
    entries.append(_bot_done(_TRUE_PRIOR_DONE_AT))
    entries.append(_bot_revert(_TRUE_REVERT_AT))

    # -- 500 further bot flap entries, none of them a Done-revert --------
    flap_base = datetime(2026, 7, 30, 0, 0, 0, tzinfo=UTC)
    for i in range(500):
        entries.append(_filler(flap_base + timedelta(minutes=i * 45)))

    if include_later_human_change:
        entries.append(
            _entry(
                created_at=_LATER_HUMAN_CHANGE_AT,
                from_state=_REVIEW,
                to_state=_PROG,
                actor_id=_HUMAN_ACTOR_ID,
            )
        )
    return entries


class FakeLinearTransport(_LinearClient):
    """`_LinearClient` with `_query` replaced by measured Linear semantics.

    Reproduces exactly what the 2026-08-27 live probe observed:

    * ``orderBy: createdAt`` orders the connection DESCENDING (newest first).
    * ``first: N`` returns the first N of that descending list — the N NEWEST.
    * ``last: N`` returns the last N of that descending list — the N OLDEST.
    * ``after: <cursor>`` resumes forward from the node whose id is the cursor.
    * ``pageInfo.endCursor`` is the last returned node's id.

    Every GraphQL round trip is recorded in ``calls`` so a test can assert
    on the actual pagination the client performed.
    """

    def __init__(
        self,
        *,
        history: list[dict[str, object]],
        comments: list[dict[str, object]] | None = None,
        issues: list[dict[str, object]] | None = None,
        team_states: list[dict[str, object]] | None = None,
    ) -> None:
        super().__init__(api_key="fake-key")
        # Descending by createdAt — Linear's own `orderBy: createdAt` order.
        self._history_desc = sorted(
            history, key=lambda n: str(n["createdAt"]), reverse=True
        )
        self._comments = comments or []
        self._issues = issues or []
        self._team_states = team_states or [
            {"id": _DONE_STATE_ID, "name": "Done", "type": "completed"},
            {"id": _STARTED_STATE_ID, "name": "In Progress", "type": "started"},
            {"id": _REVIEW_STATE_ID, "name": "In Review", "type": "started"},
        ]
        self.calls: list[dict[str, object]] = []
        self.mutations: list[tuple[str, dict[str, object]]] = []

    async def _query(  # type: ignore[override]
        self, query: str, variables: dict[str, object], timeout: float
    ) -> dict[str, object] | None:
        self.calls.append(dict(variables))
        if "history(" in query:
            return {"issue": {"history": self._history_page(variables)}}
        if "comments(" in query:
            return {"issue": {"comments": {"nodes": self._comments}}}
        if "states(" in query:
            return {"team": {"id": _TEAM_ID, "states": {"nodes": self._team_states}}}
        if "issues(" in query:
            return {
                "issues": {
                    "nodes": self._issues,
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        self.mutations.append((query, dict(variables)))
        return {"issueUpdate": {"success": True}, "commentCreate": {"success": True}}

    def _history_page(self, variables: dict[str, object]) -> dict[str, object]:
        rows = self._history_desc
        if "last" in variables:
            # Measured behavior: the tail of the DESCENDING list = oldest N.
            n = int(variables["last"])  # type: ignore[arg-type]
            page = rows[-n:] if n < len(rows) else list(rows)
            return {
                "nodes": page,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        start = 0
        after = variables.get("after")
        if after:
            ids = [str(r["id"]) for r in rows]
            start = ids.index(str(after)) + 1
        n = int(variables.get("first", 50))  # type: ignore[arg-type]
        page = rows[start : start + n]
        has_next = (start + n) < len(rows)
        return {
            "nodes": page,
            "pageInfo": {
                "hasNextPage": has_next,
                "endCursor": str(page[-1]["id"]) if page else None,
            },
        }


def _issue_stub(state_type: str = "started") -> dict[str, object]:
    return {
        "id": "issue-omn14888",
        "identifier": "OMN-14888",
        "state": {"id": _STARTED_STATE_ID, "name": "In Progress", "type": state_type},
        "team": {"id": _TEAM_ID},
    }


def _request(**overrides: object) -> ModelSyncRevertWatchdogRequest:
    defaults: dict[str, object] = {"correlation_id": uuid4(), "team_key": "OMN"}
    defaults.update(overrides)
    return ModelSyncRevertWatchdogRequest(**defaults)


@pytest.mark.unit
class TestHistoryPaginationReachesTrueEnd:
    """AC1 — the client must reach the newest entry, not the oldest page."""

    async def test_fetch_issue_history_returns_full_history_newest_included(self):
        history = _omn14888_shaped_history(include_later_human_change=True)
        transport = FakeLinearTransport(history=history)

        nodes, error = await transport.fetch_issue_history(
            "issue-omn14888", page_size=250, max_pages=20, timeout=15
        )

        assert error == ""
        assert nodes is not None
        assert len(nodes) == len(history) == 553
        created = {str(n["createdAt"]) for n in nodes}
        # The entry the truncating query could never see.
        assert _iso(_LATER_HUMAN_CHANGE_AT) in created
        assert _iso(_TRUE_REVERT_AT) in created

    async def test_pagination_walks_forward_and_stops_at_history_end(self):
        history = _omn14888_shaped_history(include_later_human_change=True)
        transport = FakeLinearTransport(history=history)

        await transport.fetch_issue_history(
            "issue-omn14888", page_size=250, max_pages=20, timeout=15
        )

        # 553 entries at 250/page = 3 round trips, never `last:`.
        assert len(transport.calls) == 3
        assert all("last" not in call for call in transport.calls)
        assert transport.calls[0].get("after") is None
        assert transport.calls[1].get("after") is not None

    async def test_max_pages_caps_the_walk(self):
        history = _omn14888_shaped_history(include_later_human_change=True)
        transport = FakeLinearTransport(history=history)

        nodes, error = await transport.fetch_issue_history(
            "issue-omn14888", page_size=100, max_pages=2, timeout=15
        )

        assert error == ""
        assert nodes is not None
        # Capped at the OLDEST end — the newest entries are still present,
        # which is what every guard in the classifier depends on.
        assert len(nodes) == 200
        created = {str(n["createdAt"]) for n in nodes}
        assert _iso(_LATER_HUMAN_CHANGE_AT) in created


@pytest.mark.unit
class TestTrueMostRecentRevertIsAnalyzed:
    """AC1/AC2 — the OMN-14888 end-to-end reproduction."""

    async def test_analyzes_true_most_recent_revert_not_the_stale_one(self):
        history = _omn14888_shaped_history(include_later_human_change=False)
        transport = FakeLinearTransport(history=history, issues=[_issue_stub()])
        handler = HandlerSyncRevertWatchdog(linear_client=transport)

        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.ticket_id == "OMN-14888"
        # The defect: `last: 50` sees only the stale 2026-07-26 revert.
        assert outcome.reverted_at == _iso(_TRUE_REVERT_AT)
        assert outcome.reverted_at != _iso(_STALE_REVERT_AT)

    async def test_later_human_state_change_outside_old_window_fires_the_guard(self):
        """AC2 — SKIPPED_STATE_CHANGED_SINCE fired 0/126 because of the window."""
        history = _omn14888_shaped_history(include_later_human_change=True)
        transport = FakeLinearTransport(history=history, issues=[_issue_stub()])
        handler = HandlerSyncRevertWatchdog(linear_client=transport)

        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.decision == (
            EnumSyncRevertWatchdogDecision.SKIPPED_STATE_CHANGED_SINCE
        )
        assert result.tickets_reflipped == 0
        assert result.reverts_detected == 1

    async def test_stale_window_verdict_would_have_been_a_reflip(self):
        """Control: the truncated window's revert IS a confident reflip candidate.

        Without this control the AC2 assertion above could pass for the
        wrong reason (e.g. the stale revert being skipped anyway). Feeding
        the handler ONLY the oldest-50 slice reproduces the defective
        verdict exactly: a would-reflip on the stale 2026-07-26 revert.
        """
        history = _omn14888_shaped_history(include_later_human_change=True)
        oldest_50 = sorted(history, key=lambda n: str(n["createdAt"]))[:50]
        transport = FakeLinearTransport(history=oldest_50, issues=[_issue_stub()])
        handler = HandlerSyncRevertWatchdog(linear_client=transport)

        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.decision == EnumSyncRevertWatchdogDecision.REFLIPPED
        assert outcome.reverted_at == _iso(_STALE_REVERT_AT)


@pytest.mark.unit
class TestPriorDoneMustBeHumanSet:
    """AC3 — the operator's restore rule, which the node never evaluated."""

    async def test_bot_set_prior_done_is_never_a_reflip_candidate(self):
        # No later human state change, no human comment — under the old
        # classifier this is an unambiguous re-flip. The pre-revert Done
        # being bot-set is the only thing standing in the way.
        history = _omn14888_shaped_history(include_later_human_change=False)
        transport = FakeLinearTransport(history=history, issues=[_issue_stub()])
        handler = HandlerSyncRevertWatchdog(linear_client=transport)

        result = await handler.handle(_request(apply=True))

        outcome = result.outcomes[0]
        assert outcome.decision == (
            EnumSyncRevertWatchdogDecision.SKIPPED_PRIOR_DONE_NOT_HUMAN_SET
        )
        assert outcome.prior_done_actor_kind == EnumPriorDoneActorKind.BOT
        assert outcome.prior_done_at == _iso(_TRUE_PRIOR_DONE_AT)
        assert outcome.applied is False
        assert result.tickets_reflipped == 0
        assert transport.mutations == []

    async def test_human_set_prior_done_still_reflips(self):
        """The precondition must not disable the correction it guards."""
        revert_at = datetime(2026, 8, 24, 13, 45, 21, 340000, tzinfo=UTC)
        history = [
            _human_done(revert_at - timedelta(days=4)),
            _bot_revert(revert_at),
        ]
        transport = FakeLinearTransport(history=history, issues=[_issue_stub()])
        handler = HandlerSyncRevertWatchdog(linear_client=transport)

        result = await handler.handle(_request(apply=True))

        outcome = result.outcomes[0]
        assert outcome.decision == EnumSyncRevertWatchdogDecision.REFLIPPED
        assert outcome.prior_done_actor_kind == EnumPriorDoneActorKind.HUMAN
        assert outcome.applied is True

    async def test_indeterminate_prior_done_fails_closed(self):
        """No transition INTO the prior completed state anywhere in history."""
        revert_at = datetime(2026, 8, 24, 13, 45, 21, 340000, tzinfo=UTC)
        transport = FakeLinearTransport(
            history=[_bot_revert(revert_at)], issues=[_issue_stub()]
        )
        handler = HandlerSyncRevertWatchdog(linear_client=transport)

        result = await handler.handle(_request(apply=True))

        outcome = result.outcomes[0]
        assert outcome.decision == (
            EnumSyncRevertWatchdogDecision.SKIPPED_PRIOR_DONE_NOT_HUMAN_SET
        )
        assert outcome.prior_done_actor_kind == EnumPriorDoneActorKind.UNKNOWN
        assert transport.mutations == []

    async def test_prior_done_check_precedes_the_comment_fetch(self):
        """The guard is pure-history — it must not cost an extra Linear call."""
        history = _omn14888_shaped_history(include_later_human_change=False)
        transport = FakeLinearTransport(history=history, issues=[_issue_stub()])
        handler = HandlerSyncRevertWatchdog(linear_client=transport)

        await handler.handle(_request())

        assert transport.mutations == []
        # issues + 3 history pages, and no comments round trip.
        assert len(transport.calls) == 4
