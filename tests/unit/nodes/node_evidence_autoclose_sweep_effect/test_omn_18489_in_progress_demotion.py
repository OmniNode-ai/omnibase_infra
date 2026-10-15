# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18489 AC1-AC4: the In Progress demotion leg and its five conditions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.in_progress_demotion import (
    InProgressDemotionLeg,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_in_progress_demotion_request import (
    ModelInProgressDemotionRequest,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_handler_evidence_autoclose_sweep import (
    _request as _sweep_request,
)

pytestmark = pytest.mark.unit

_NOW = datetime.now(tz=UTC)
_OLD = (_NOW - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
_RECENT = (_NOW - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
_SWEEP_COMMENT = (
    "<!-- onex:in-progress-probe-hygiene:no-executable-probe -->\nsaid once"
)


def _issue(
    ticket: str = "OMN-9001",
    *,
    state_name: str = "In Progress",
    state_type: str = "started",
    started_at: str | None = _OLD,
    comments: list[dict[str, str]] | None = None,
    comments_next_page: bool = False,
) -> dict[str, object]:
    return {
        "id": f"id-{ticket}",
        "identifier": ticket,
        "startedAt": started_at,
        "state": {"id": "state-in-progress", "name": state_name, "type": state_type},
        "team": {"id": "team-1"},
        "comments": {
            "pageInfo": {"hasNextPage": comments_next_page},
            "nodes": comments or [],
        },
    }


class _Linear:
    def __init__(
        self,
        issues: list[dict[str, object]],
        *,
        enumeration_error: str = "",
        backlog_state_id: str | None = "state-backlog",
        write_ok: bool = True,
        readback_moves: bool = True,
        comment_ok: bool = True,
    ) -> None:
        self._issues = issues
        self._enumeration_error = enumeration_error
        self._backlog_state_id = backlog_state_id
        self._write_ok = write_ok
        self._readback_moves = readback_moves
        self._comment_ok = comment_ok
        self.since_isos: list[str] = []
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_in_progress_issues(
        self, since_iso: str
    ) -> tuple[list[dict[str, object]] | None, str]:
        self.since_isos.append(since_iso)
        if self._enumeration_error:
            return None, self._enumeration_error
        return self._issues, ""

    async def fetch_backlog_state_id(self, team_id: str) -> str | None:
        return self._backlog_state_id

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return self._write_ok

    async def fetch_issue(self, ticket_id: str) -> dict[str, object] | None:
        moved = bool(self.state_updates) and self._readback_moves
        return {"state": {"type": "backlog" if moved else "started", "name": "Backlog"}}

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return self._comment_ok


def _gh(
    open_prs: list[dict[str, object]] | None = None, *, error: str = "", total=None
):
    calls: list[list[str]] = []
    items = open_prs or []

    async def run(args: list[str], timeout: float):
        calls.append(args)
        if error:
            return None, error
        return {
            "total_count": len(items) if total is None else total,
            "incomplete_results": False,
            "items": items,
        }, ""

    run.calls = calls
    return run


def _pr(number: int, title: str, body: str = "") -> dict[str, object]:
    return {"number": number, "title": title, "body": body}


def _request(**overrides: object) -> ModelInProgressDemotionRequest:
    values: dict[str, object] = {
        "correlation_id": uuid4(),
        "window_hours": 168,
        "apply": True,
        "max_demotions_per_run": 5,
        "pr_owner": "OmniNode-ai",
        "gh_timeout_seconds": 30,
    }
    values.update(overrides)
    return ModelInProgressDemotionRequest(**values)


def _handler(linear: _Linear, gh=None) -> InProgressDemotionLeg:
    return InProgressDemotionLeg(linear_client=linear, run_gh_command=gh or _gh())


async def test_ticket_meeting_all_conditions_is_demoted_with_a_comment():
    linear = _Linear([_issue(comments=[{"createdAt": _OLD, "body": _SWEEP_COMMENT}])])
    result = await _handler(linear).handle(_request())

    assert [o.decision.value for o in result.outcomes] == ["demoted"]
    assert result.tickets_demoted == 1
    assert linear.state_updates == [("id-OMN-9001", "state-backlog")]
    assert len(linear.comments) == 1
    body = linear.comments[0][1]
    assert "168" in body
    assert "OMN-9001" in body
    for resolution in ("In Progress", "comment", "pull request", "claim", "fence"):
        assert resolution in body
    assert "<!-- onex:in-progress-demotion" in body


async def test_dry_run_reports_would_demote_and_writes_nothing():
    linear = _Linear([_issue()])
    result = await _handler(linear).handle(_request(apply=False))
    assert result.outcomes[0].decision.value == "would_demote"
    assert result.dry_run is True
    assert result.tickets_would_demote == 1
    assert linear.state_updates == []
    assert linear.comments == []


@pytest.mark.parametrize(
    ("issue", "gh", "exclude", "expected"),
    [
        (
            _issue(state_name="In Review"),
            _gh(),
            (),
            "skipped_not_in_progress",
        ),
        (
            _issue(comments=[{"createdAt": _RECENT, "body": "still on it"}]),
            _gh(),
            (),
            "skipped_recent_comment",
        ),
        (
            _issue(),
            _gh([_pr(7, "feat(OMN-9001): the work")]),
            (),
            "skipped_open_pull_request",
        ),
        (
            _issue(
                comments=[
                    {"createdAt": _RECENT, "body": "actor: lane-x (sonnet)\nclaiming"}
                ]
            ),
            _gh(),
            (),
            "skipped_live_claim",
        ),
        (_issue(), _gh(), ("omn-9001",), "skipped_excluded"),
    ],
    ids=["state", "comment", "pull-request", "claim", "fence"],
)
async def test_each_failed_condition_alone_withholds_the_demotion(
    issue, gh, exclude, expected
):
    linear = _Linear([issue])
    result = await _handler(linear, gh).handle(_request(exclude_tickets=exclude))
    assert result.outcomes[0].decision.value == expected
    assert linear.state_updates == []
    assert linear.comments == []
    assert result.tickets_skipped == 1


async def test_a_sweeps_own_comment_inside_the_window_does_not_count_as_activity():
    linear = _Linear(
        [_issue(comments=[{"createdAt": _RECENT, "body": _SWEEP_COMMENT}])]
    )
    result = await _handler(linear).handle(_request())
    assert result.outcomes[0].decision.value == "demoted"


async def test_ticket_started_inside_the_window_is_not_demoted():
    linear = _Linear([_issue(started_at=_RECENT)])
    result = await _handler(linear).handle(_request())
    assert result.outcomes[0].decision.value == "skipped_started_within_window"
    assert linear.state_updates == []


@pytest.mark.parametrize("started_at", [None, "not-a-date"])
async def test_unreadable_start_time_is_an_error_not_a_demotion(started_at):
    linear = _Linear([_issue(started_at=started_at)])
    result = await _handler(linear).handle(_request())
    assert result.outcomes[0].decision.value == "error_linear_api"
    assert linear.state_updates == []


async def test_a_full_comment_page_is_read_as_recent_activity():
    linear = _Linear([_issue(comments_next_page=True)])
    result = await _handler(linear).handle(_request())
    assert result.outcomes[0].decision.value == "skipped_recent_comment"


async def test_unreadable_comment_connection_is_an_error_not_a_demotion():
    broken = _issue()
    broken["comments"] = None
    linear = _Linear([broken])
    result = await _handler(linear).handle(_request())
    assert result.outcomes[0].decision.value == "error_linear_api"
    assert linear.state_updates == []


async def test_pull_request_resolution_failure_leaves_the_ticket_untouched():
    linear = _Linear([_issue()])
    result = await _handler(linear, _gh(error="HTTP 502")).handle(_request())
    outcome = result.outcomes[0]
    assert outcome.decision.value == "error_github_api"
    assert "HTTP 502" in outcome.reason
    assert linear.state_updates == []
    assert result.tickets_errored == 1


async def test_incomplete_pull_request_search_is_an_error_not_an_absence():
    linear = _Linear([_issue()])
    result = await _handler(linear, _gh([_pr(1, "other")], total=5000)).handle(
        _request()
    )
    assert result.outcomes[0].decision.value == "error_github_api"
    assert linear.state_updates == []


async def test_pull_request_citing_a_longer_ticket_id_does_not_hold_this_ticket():
    linear = _Linear([_issue()])
    result = await _handler(linear, _gh([_pr(8, "feat(OMN-90011): unrelated")])).handle(
        _request()
    )
    assert result.outcomes[0].decision.value == "demoted"


async def test_pull_request_body_citation_holds_the_ticket():
    linear = _Linear([_issue()])
    gh = _gh([_pr(9, "chore: tidy", body="Part of OMN-9001.")])
    result = await _handler(linear, gh).handle(_request())
    assert result.outcomes[0].decision.value == "skipped_open_pull_request"


async def test_open_pull_requests_are_enumerated_once_per_pass():
    linear = _Linear([_issue("OMN-9001"), _issue("OMN-9002")])
    gh = _gh()
    await _handler(linear, gh).handle(_request())
    assert len(gh.calls) == 1


async def test_enumeration_failure_judges_nothing():
    linear = _Linear([_issue()], enumeration_error="Linear 500")
    result = await _handler(linear).handle(_request())
    assert result.success is False
    assert "Linear 500" in result.error_message
    assert result.outcomes == ()
    assert linear.state_updates == []


async def test_state_write_failure_is_an_error():
    linear = _Linear([_issue()], write_ok=False)
    result = await _handler(linear).handle(_request())
    assert result.outcomes[0].decision.value == "error_linear_api"
    assert linear.comments == []


async def test_missing_backlog_state_is_an_error_and_writes_nothing():
    linear = _Linear([_issue()], backlog_state_id=None)
    result = await _handler(linear).handle(_request())
    assert result.outcomes[0].decision.value == "error_linear_api"
    assert linear.state_updates == []


async def test_unconfirmed_readback_is_an_error_not_a_demotion():
    linear = _Linear([_issue()], readback_moves=False)
    result = await _handler(linear).handle(_request())
    assert result.outcomes[0].decision.value == "error_readback_unconfirmed"
    assert result.tickets_demoted == 0


async def test_comment_failure_after_the_write_is_reported_not_hidden():
    linear = _Linear([_issue()], comment_ok=False)
    result = await _handler(linear).handle(_request())
    assert result.outcomes[0].decision.value == "error_comment_unposted"
    assert result.tickets_errored == 1


async def test_budget_bounds_the_state_writes_in_one_pass():
    linear = _Linear([_issue("OMN-9001"), _issue("OMN-9002"), _issue("OMN-9003")])
    result = await _handler(linear).handle(_request(max_demotions_per_run=2))
    decisions = [o.decision.value for o in result.outcomes]
    assert decisions == ["demoted", "demoted", "skipped_budget_exhausted"]
    assert len(linear.state_updates) == 2


async def test_two_windows_demote_different_fixture_sets():
    ten_days = (_NOW - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    issues = [
        _issue("OMN-9001", started_at=ten_days),
        _issue("OMN-9002", started_at=_OLD),
    ]
    short = await _handler(_Linear(issues)).handle(_request(window_hours=24 * 7))
    long = await _handler(_Linear(issues)).handle(_request(window_hours=24 * 20))
    assert {
        o.ticket_identifier for o in short.outcomes if o.decision.value == "demoted"
    } == {
        "OMN-9001",
        "OMN-9002",
    }
    assert {
        o.ticket_identifier for o in long.outcomes if o.decision.value == "demoted"
    } == {"OMN-9002"}


async def test_the_window_has_no_default():
    with pytest.raises(ValueError):
        ModelInProgressDemotionRequest(
            correlation_id=uuid4(),
            apply=True,
            max_demotions_per_run=1,
            pr_owner="OmniNode-ai",
            gh_timeout_seconds=30,
        )


async def test_the_comment_window_sent_to_linear_matches_the_request():
    linear = _Linear([_issue()])
    await _handler(linear).handle(_request(window_hours=48))
    since = datetime.strptime(linear.since_isos[0], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC
    )
    assert abs((datetime.now(tz=UTC) - since) - timedelta(hours=48)) < timedelta(
        minutes=1
    )


# -- wiring into the sweep handler -------------------------------------------


class _SweepLinear(_Linear):
    """Adds the closer-side calls the sweep makes while enumerating companions."""

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return ()


def _sweep(linear: _SweepLinear, open_prs=None) -> HandlerEvidenceAutocloseSweep:
    async def run_gh(args: list[str], timeout: float):
        path = args[2]
        if path.startswith("search/issues?q=org:"):
            return {
                "total_count": len(open_prs or []),
                "incomplete_results": False,
                "items": open_prs or [],
            }, ""
        return [], ""  # no merged companions in the window

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,
        autoclose_disabled=False,
        run_gh_command=run_gh,
    )


async def test_demotion_leg_is_off_when_the_window_is_zero():
    linear = _SweepLinear([_issue()])
    result = await _sweep(linear).handle(_sweep_request(apply=True))
    assert result.in_progress_demotion is None
    assert linear.since_isos == []
    assert linear.state_updates == []


async def test_sweep_runs_the_leg_inside_the_same_tick_when_a_window_is_declared():
    linear = _SweepLinear([_issue()])
    result = await _sweep(linear).handle(
        _sweep_request(apply=True, demotion_window_hours=168)
    )
    assert result.in_progress_demotion is not None
    assert result.in_progress_demotion.tickets_demoted == 1
    assert linear.state_updates == [("id-OMN-9001", "state-backlog")]


async def test_sweep_dry_run_demotes_nothing():
    linear = _SweepLinear([_issue()])
    result = await _sweep(linear).handle(
        _sweep_request(apply=False, demotion_window_hours=168)
    )
    assert result.in_progress_demotion is not None
    assert result.in_progress_demotion.tickets_would_demote == 1
    assert linear.state_updates == []


async def test_sweep_exclusion_fence_reaches_the_leg():
    linear = _SweepLinear([_issue()])
    result = await _sweep(linear).handle(
        _sweep_request(
            apply=True, demotion_window_hours=168, exclude_tickets=("OMN-9001",)
        )
    )
    assert result.in_progress_demotion is not None
    assert result.in_progress_demotion.outcomes[0].decision.value == "skipped_excluded"
    assert linear.state_updates == []


async def test_sweep_disarmed_run_demotes_nothing():
    linear = _SweepLinear([_issue()])
    result = await _sweep(linear).handle(
        _sweep_request(
            apply=True, demotion_window_hours=168, disarmed_by_ticket="OMN-1"
        )
    )
    assert result.in_progress_demotion is not None
    assert result.in_progress_demotion.dry_run is True
    assert linear.state_updates == []


async def test_sweep_kill_switch_performs_no_demotion_io():
    linear = _SweepLinear([_issue()])
    handler = HandlerEvidenceAutocloseSweep(
        linear_client=linear,
        autoclose_disabled=True,
    )
    result = await handler.handle(_sweep_request(apply=True, demotion_window_hours=168))
    assert result.kill_switch_engaged is True
    assert result.in_progress_demotion is None
    assert linear.since_isos == []
