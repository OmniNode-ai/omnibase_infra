# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for HandlerSyncRevertWatchdog — recorded stateHistory fixtures.

Covers OMN-16536's sync-revert watchdog: detection via Linear's own
actorId/botActor automation signature, the no-human-comment-nearby guard,
the no-later-human-state-change guard (scoped to state-change entries
only), the already-resolved short-circuit, unresolvable-target fail
closed, every Linear-API failure path, the kill switch, and two
incident-shaped end-to-end fixtures reproducing the OMN-15977 (bare
`Refs:` mention automation-fire) and OMN-15751 (draft->ready_for_review
automation-fire) revert signatures from OMN-16536's own body/comments.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_sync_revert_watchdog_effect.handlers.handler_sync_revert_watchdog import (
    HandlerSyncRevertWatchdog,
    _parse_iso,
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


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _history_entry(
    *,
    created_at: datetime,
    from_type: str = "",
    to_type: str = "",
    from_name: str = "Done",
    to_name: str = "In Progress",
    from_id: str = _DONE_STATE_ID,
    to_id: str = _STARTED_STATE_ID,
    actor_id: str | None = None,
    bot_type: str = "",
) -> dict[str, object]:
    return {
        "id": str(uuid4()),
        "createdAt": _iso(created_at),
        "actorId": actor_id,
        "botActor": {"type": bot_type, "name": "GitHub"} if bot_type else None,
        "fromState": (
            {"id": from_id, "name": from_name, "type": from_type} if from_type else None
        ),
        "toState": (
            {"id": to_id, "name": to_name, "type": to_type} if to_type else None
        ),
    }


def _automation_revert_entry(
    created_at: datetime, **overrides: object
) -> dict[str, object]:
    defaults: dict[str, object] = {
        "created_at": created_at,
        "from_type": "completed",
        "to_type": "started",
        "actor_id": None,
        "bot_type": "github",
    }
    defaults.update(overrides)
    return _history_entry(**defaults)  # type: ignore[arg-type]


def _human_set_done_entry(created_at: datetime) -> dict[str, object]:
    """A human moving the ticket INTO Done.

    OMN-16762 added a restore precondition: the watchdog only re-flips to
    a completed state a HUMAN set. Every fixture below whose expected
    decision is downstream of that gate therefore needs an explicit
    human-set Done in its history — previously these fixtures carried a
    bare revert entry with no transition into Done at all, which now
    resolves to EnumPriorDoneActorKind.UNKNOWN and fails closed.
    """
    return _history_entry(
        created_at=created_at,
        from_type="started",
        to_type="completed",
        from_name="In Progress",
        to_name="Done",
        from_id=_STARTED_STATE_ID,
        to_id=_DONE_STATE_ID,
        actor_id="real-user-id",
    )


def _issue_stub(
    *,
    issue_id: str = "issue-1",
    identifier: str = "OMN-9999",
    state_type: str = "started",
) -> dict[str, object]:
    return {
        "id": issue_id,
        "identifier": identifier,
        "state": {"id": _STARTED_STATE_ID, "name": "In Progress", "type": state_type},
        "team": {"id": _TEAM_ID},
    }


def _comment(
    *, created_at: datetime, is_human: bool = True, body: str = "some comment"
) -> dict[str, object]:
    return {
        "id": str(uuid4()),
        "createdAt": _iso(created_at),
        "body": body,
        "user": {"id": "user-1"} if is_human else None,
    }


class FakeLinearClient:
    """Fake Linear client — canned issue/history/comment payloads, records mutations."""

    def __init__(
        self,
        *,
        issues: list[dict[str, object]] | None = None,
        history_by_issue: dict[str, list[dict[str, object]]] | None = None,
        comments_by_issue: dict[str, list[dict[str, object]]] | None = None,
        team_states: list[dict[str, object]] | None = None,
        enum_error: str = "",
        history_error_by_issue: dict[str, str] | None = None,
        comments_error_by_issue: dict[str, str] | None = None,
        update_state_result: bool = True,
        create_comment_result: bool = True,
    ) -> None:
        self._issues = issues or []
        self._history_by_issue = history_by_issue or {}
        self._comments_by_issue = comments_by_issue or {}
        self._team_states = (
            team_states
            if team_states is not None
            else [
                {"id": _DONE_STATE_ID, "name": "Done", "type": "completed"},
                {"id": _STARTED_STATE_ID, "name": "In Progress", "type": "started"},
            ]
        )
        self._enum_error = enum_error
        self._history_error_by_issue = history_error_by_issue or {}
        self._comments_error_by_issue = comments_error_by_issue or {}
        self._update_state_result = update_state_result
        self._create_comment_result = create_comment_result
        self.state_updates: list[tuple[str, str]] = []
        self.comments_posted: list[tuple[str, str]] = []
        self.history_calls: list[str] = []
        self.comments_calls: list[str] = []

    async def fetch_recently_updated_issues(
        self, team_key, since_iso, max_issues, timeout
    ):
        if self._enum_error:
            return [], self._enum_error
        return self._issues[:max_issues], ""

    async def fetch_issue_history(self, issue_id, page_size, max_pages, timeout):
        self.history_calls.append(issue_id)
        if issue_id in self._history_error_by_issue:
            return None, self._history_error_by_issue[issue_id]
        return self._history_by_issue.get(issue_id, []), ""

    async def fetch_issue_comments(self, issue_id, timeout):
        self.comments_calls.append(issue_id)
        if issue_id in self._comments_error_by_issue:
            return None, self._comments_error_by_issue[issue_id]
        return self._comments_by_issue.get(issue_id, []), ""

    async def fetch_team_states(self, team_id, timeout):
        return self._team_states

    async def update_issue_state(self, issue_id, state_id, timeout):
        self.state_updates.append((issue_id, state_id))
        return self._update_state_result

    async def create_comment(self, issue_id, body, timeout):
        self.comments_posted.append((issue_id, body))
        return self._create_comment_result


def _request(**overrides: object) -> ModelSyncRevertWatchdogRequest:
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "team_key": "OMN",
        "lookback_hours": 24,
        "apply": False,
    }
    defaults.update(overrides)
    return ModelSyncRevertWatchdogRequest(**defaults)


@pytest.mark.unit
class TestParseIso:
    def test_parses_z_suffixed_timestamp(self):
        parsed = _parse_iso("2026-08-24T13:45:21.340Z")
        assert parsed is not None
        assert parsed.year == 2026
        assert parsed.tzinfo is not None

    def test_none_on_garbage(self):
        assert _parse_iso("not-a-timestamp") is None
        assert _parse_iso(None) is None
        assert _parse_iso(12345) is None


@pytest.mark.unit
class TestKillSwitch:
    async def test_kill_switch_env_var_short_circuits(self, monkeypatch):
        monkeypatch.setenv("ONEX_SYNC_REVERT_WATCHDOG_DISABLED", "1")
        linear = FakeLinearClient(issues=[_issue_stub()])
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request())
        assert result.kill_switch_engaged is True
        assert result.issues_scanned == 0
        assert linear.history_calls == []

    async def test_kill_switch_constructor_override(self):
        linear = FakeLinearClient(issues=[_issue_stub()])
        handler = HandlerSyncRevertWatchdog(
            linear_client=linear, kill_switch_disabled=True
        )
        result = await handler.handle(_request())
        assert result.kill_switch_engaged is True
        assert linear.history_calls == []


@pytest.mark.unit
class TestNoRevertOrAlreadyResolved:
    async def test_no_completed_to_noncompleted_transition_is_skipped(self):
        now = _now()
        issue = _issue_stub(state_type="started")
        history = [
            _history_entry(
                created_at=now - timedelta(hours=1),
                from_type="unstarted",
                to_type="started",
                actor_id="user-1",
            )
        ]
        linear = FakeLinearClient(issues=[issue], history_by_issue={"issue-1": history})
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request())
        assert result.outcomes[0].decision == (
            EnumSyncRevertWatchdogDecision.SKIPPED_NO_REVERT_FOUND
        )
        assert result.reverts_detected == 0

    async def test_already_resolved_current_state_completed(self):
        now = _now()
        issue = _issue_stub(state_type="completed")
        history = [
            _human_set_done_entry(now - timedelta(days=1)),
            _automation_revert_entry(now - timedelta(seconds=5)),
        ]
        linear = FakeLinearClient(issues=[issue], history_by_issue={"issue-1": history})
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request())
        assert result.outcomes[0].decision == (
            EnumSyncRevertWatchdogDecision.SKIPPED_ALREADY_RESOLVED
        )
        # A revert transition existed even though nothing needed doing.
        assert result.reverts_detected == 1
        assert result.tickets_reflipped == 0


@pytest.mark.unit
class TestHumanActorSkips:
    async def test_human_actored_revert_is_out_of_scope(self):
        now = _now()
        issue = _issue_stub(state_type="started")
        history = [
            _history_entry(
                created_at=now - timedelta(seconds=5),
                from_type="completed",
                to_type="started",
                actor_id="real-user-id",
            )
        ]
        linear = FakeLinearClient(issues=[issue], history_by_issue={"issue-1": history})
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request())
        assert result.outcomes[0].decision == (
            EnumSyncRevertWatchdogDecision.SKIPPED_HUMAN_ACTOR
        )
        assert linear.comments_calls == []  # never needed to check comments

    async def test_later_human_state_change_defers(self):
        now = _now()
        issue = _issue_stub(state_type="started")
        history = [
            _automation_revert_entry(now - timedelta(hours=2)),
            _history_entry(
                created_at=now - timedelta(hours=1),
                from_type="started",
                to_type="unstarted",
                to_name="Todo",
                actor_id="real-user-id",
            ),
        ]
        linear = FakeLinearClient(issues=[issue], history_by_issue={"issue-1": history})
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request())
        assert result.outcomes[0].decision == (
            EnumSyncRevertWatchdogDecision.SKIPPED_STATE_CHANGED_SINCE
        )

    async def test_later_human_non_state_edit_does_not_suppress(self):
        """A human touching an unrelated field (no toState) must not mask a real revert."""
        now = _now()
        issue = _issue_stub(state_type="started")
        history = [
            _human_set_done_entry(now - timedelta(days=1)),
            _automation_revert_entry(now - timedelta(hours=2)),
            {
                "id": str(uuid4()),
                "createdAt": _iso(now - timedelta(hours=1)),
                "actorId": "real-user-id",
                "botActor": None,
                "fromState": None,
                "toState": None,
                "addedLabelIds": ["label-1"],
            },
        ]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-1": history},
            comments_by_issue={"issue-1": []},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request(apply=True))
        assert result.outcomes[0].decision == EnumSyncRevertWatchdogDecision.REFLIPPED
        assert linear.state_updates == [("issue-1", _DONE_STATE_ID)]


@pytest.mark.unit
class TestHumanCommentWindow:
    async def test_human_comment_inside_window_is_treated_as_explained(self):
        now = _now()
        revert_at = now - timedelta(minutes=10)
        issue = _issue_stub(state_type="started")
        history = [
            _human_set_done_entry(revert_at - timedelta(days=1)),
            _automation_revert_entry(revert_at),
        ]
        comments = [
            _comment(created_at=revert_at + timedelta(minutes=2), is_human=True)
        ]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-1": history},
            comments_by_issue={"issue-1": comments},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request(human_comment_window_seconds=3600))
        assert result.outcomes[0].decision == (
            EnumSyncRevertWatchdogDecision.SKIPPED_HUMAN_COMMENT_NEARBY
        )

    async def test_human_comment_outside_window_does_not_suppress(self):
        now = _now()
        revert_at = now - timedelta(hours=5)
        issue = _issue_stub(state_type="started")
        history = [
            _human_set_done_entry(revert_at - timedelta(days=1)),
            _automation_revert_entry(revert_at),
        ]
        comments = [_comment(created_at=revert_at + timedelta(hours=3), is_human=True)]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-1": history},
            comments_by_issue={"issue-1": comments},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(
            _request(apply=True, human_comment_window_seconds=3600)
        )
        assert result.outcomes[0].decision == EnumSyncRevertWatchdogDecision.REFLIPPED

    async def test_bot_comment_inside_window_does_not_suppress(self):
        """Only a HUMAN (user-attributed) comment counts — a bot comment must not."""
        now = _now()
        revert_at = now - timedelta(seconds=30)
        issue = _issue_stub(state_type="started")
        history = [
            _human_set_done_entry(revert_at - timedelta(days=1)),
            _automation_revert_entry(revert_at),
        ]
        comments = [_comment(created_at=revert_at, is_human=False, body="bot note")]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-1": history},
            comments_by_issue={"issue-1": comments},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request(apply=True))
        assert result.outcomes[0].decision == EnumSyncRevertWatchdogDecision.REFLIPPED

    async def test_watchdogs_own_prior_comment_is_excluded_from_the_check(self):
        """A prior watchdog diagnosis comment must never be mistaken for a human explanation."""
        now = _now()
        revert_at = now - timedelta(minutes=5)
        issue = _issue_stub(state_type="started")
        history = [
            _human_set_done_entry(revert_at - timedelta(days=1)),
            _automation_revert_entry(revert_at),
        ]
        marker = "sync-revert-watchdog (OMN-16536)"
        comments = [
            _comment(
                created_at=revert_at + timedelta(seconds=1),
                is_human=True,
                body=f"Automatic re-flip — {marker}. Detected an automation revert.",
            )
        ]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-1": history},
            comments_by_issue={"issue-1": comments},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(
            _request(apply=True, watchdog_comment_marker=marker)
        )
        assert result.outcomes[0].decision == EnumSyncRevertWatchdogDecision.REFLIPPED


@pytest.mark.unit
class TestStateResolution:
    async def test_target_state_no_longer_resolvable_fails_closed(self):
        now = _now()
        issue = _issue_stub(state_type="started")
        history = [
            _human_set_done_entry(now - timedelta(days=1)),
            _automation_revert_entry(now - timedelta(seconds=5)),
        ]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-1": history},
            comments_by_issue={"issue-1": []},
            team_states=[  # Done state has been deleted/renamed off the team.
                {"id": _STARTED_STATE_ID, "name": "In Progress", "type": "started"}
            ],
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request(apply=True))
        assert result.outcomes[0].decision == (
            EnumSyncRevertWatchdogDecision.ERROR_STATE_NOT_RESOLVABLE
        )
        assert linear.state_updates == []

    async def test_dry_run_never_checks_team_states_or_mutates(self):
        now = _now()
        issue = _issue_stub(state_type="started")
        history = [
            _human_set_done_entry(now - timedelta(days=1)),
            _automation_revert_entry(now - timedelta(seconds=5)),
        ]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-1": history},
            comments_by_issue={"issue-1": []},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request(apply=False))
        assert result.dry_run is True
        assert result.outcomes[0].decision == EnumSyncRevertWatchdogDecision.REFLIPPED
        assert result.outcomes[0].applied is False
        assert linear.state_updates == []
        assert linear.comments_posted == []


@pytest.mark.unit
class TestLinearApiFailures:
    async def test_enumeration_failure_is_sweep_level(self):
        linear = FakeLinearClient(enum_error="rate limited")
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request())
        assert result.success is False
        assert "rate limited" in result.error_message
        assert result.outcomes == ()

    async def test_history_fetch_failure_is_per_ticket_error(self):
        issue = _issue_stub()
        linear = FakeLinearClient(
            issues=[issue],
            history_error_by_issue={"issue-1": "boom"},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request())
        assert result.success is True  # per-ticket failure never aborts the sweep
        assert (
            result.outcomes[0].decision
            == EnumSyncRevertWatchdogDecision.ERROR_LINEAR_API
        )

    async def test_comments_fetch_failure_is_per_ticket_error(self):
        now = _now()
        issue = _issue_stub(state_type="started")
        history = [
            _human_set_done_entry(now - timedelta(days=1)),
            _automation_revert_entry(now - timedelta(seconds=5)),
        ]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-1": history},
            comments_error_by_issue={"issue-1": "boom"},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request())
        assert (
            result.outcomes[0].decision
            == EnumSyncRevertWatchdogDecision.ERROR_LINEAR_API
        )

    async def test_mutation_failure_is_per_ticket_error(self):
        now = _now()
        issue = _issue_stub(state_type="started")
        history = [
            _human_set_done_entry(now - timedelta(days=1)),
            _automation_revert_entry(now - timedelta(seconds=5)),
        ]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-1": history},
            comments_by_issue={"issue-1": []},
            update_state_result=False,
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request(apply=True))
        assert (
            result.outcomes[0].decision
            == EnumSyncRevertWatchdogDecision.ERROR_LINEAR_API
        )


@pytest.mark.unit
class TestIncidentShapedFixtures:
    """End-to-end reproductions of the two firing surfaces OMN-16536 documents."""

    async def test_omn15977_bare_refs_mention_signature(self):
        """OMN-15977: Done -> reverted 3.34s after an unrelated PR's bare Refs: mention.

        Real incident timestamps from OMN-16536's body: PR merged
        2026-08-24T13:45:18Z, ticket reverted 2026-08-24T13:45:21.340Z.
        """
        revert_at = datetime(2026, 8, 24, 13, 45, 21, 340000, tzinfo=UTC)
        issue = _issue_stub(
            issue_id="issue-omn15977", identifier="OMN-15977", state_type="started"
        )
        history = [
            _history_entry(
                created_at=datetime(2026, 8, 20, 9, 0, 0, tzinfo=UTC),
                from_type="started",
                to_type="completed",
                from_name="In Progress",
                to_name="Done",
                actor_id="real-user-id",
            ),
            _automation_revert_entry(revert_at, from_name="Done", to_name="In Review"),
        ]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-omn15977": history},
            comments_by_issue={"issue-omn15977": []},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request(apply=True, team_key="OMN"))

        outcome = result.outcomes[0]
        assert outcome.ticket_id == "OMN-15977"
        assert outcome.decision == EnumSyncRevertWatchdogDecision.REFLIPPED
        assert outcome.bot_actor_type == "github"
        assert outcome.from_state_name == "Done"
        assert outcome.to_state_name == "In Review"
        assert linear.state_updates == [("issue-omn15977", _DONE_STATE_ID)]
        assert len(linear.comments_posted) == 1
        posted_issue_id, posted_body = linear.comments_posted[0]
        assert posted_issue_id == "issue-omn15977"
        assert "OMN-16536" in posted_body
        assert result.tickets_reflipped == 1
        assert result.reverts_detected == 1

    async def test_omn15751_ready_for_review_signature(self):
        """OMN-15751: Done for ~53h, reverted 2.56s after a ready_for_review lifecycle
        event on the ticket's own draft PR -- same botActor(github)/actorId-null
        signature as a Refs: mention, confirming detection is signature-based, not
        tied to any one specific GitHub trigger event.
        """
        done_at = datetime(2026, 8, 22, 21, 9, 44, tzinfo=UTC)
        revert_at = datetime(2026, 8, 25, 2, 13, 34, 560000, tzinfo=UTC)
        issue = _issue_stub(
            issue_id="issue-omn15751", identifier="OMN-15751", state_type="started"
        )
        history = [
            _history_entry(
                created_at=done_at,
                from_type="started",
                to_type="completed",
                from_name="In Progress",
                to_name="Done",
                actor_id="real-user-id",
            ),
            _automation_revert_entry(
                revert_at, from_name="Done", to_name="In Progress"
            ),
        ]
        linear = FakeLinearClient(
            issues=[issue],
            history_by_issue={"issue-omn15751": history},
            comments_by_issue={"issue-omn15751": []},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request(apply=True))

        outcome = result.outcomes[0]
        assert outcome.ticket_id == "OMN-15751"
        assert outcome.decision == EnumSyncRevertWatchdogDecision.REFLIPPED
        assert linear.state_updates == [("issue-omn15751", _DONE_STATE_ID)]


@pytest.mark.unit
class TestSummaryCounts:
    async def test_run_level_counts_across_mixed_outcomes(self):
        now = _now()
        issues = [
            _issue_stub(issue_id="i1", identifier="OMN-1", state_type="started"),
            _issue_stub(issue_id="i2", identifier="OMN-2", state_type="started"),
            _issue_stub(issue_id="i3", identifier="OMN-3", state_type="unstarted"),
        ]
        history_by_issue = {
            "i1": [
                _human_set_done_entry(now - timedelta(days=1)),
                _automation_revert_entry(now - timedelta(seconds=10)),
            ],
            "i2": [
                _history_entry(
                    created_at=now - timedelta(seconds=10),
                    from_type="completed",
                    to_type="started",
                    actor_id="real-user-id",
                )
            ],
            "i3": [
                _history_entry(
                    created_at=now - timedelta(hours=1),
                    from_type="unstarted",
                    to_type="started",
                    actor_id="real-user-id",
                )
            ],
        }
        linear = FakeLinearClient(
            issues=issues,
            history_by_issue=history_by_issue,
            comments_by_issue={"i1": [], "i2": [], "i3": []},
        )
        handler = HandlerSyncRevertWatchdog(linear_client=linear)
        result = await handler.handle(_request(apply=True))

        assert result.issues_scanned == 3
        assert result.reverts_detected == 2  # i1 (automation) + i2 (human actor)
        assert result.tickets_reflipped == 1  # only i1
        assert result.tickets_skipped == 2  # i2 human-actor, i3 no-revert-found
        assert result.tickets_errored == 0
