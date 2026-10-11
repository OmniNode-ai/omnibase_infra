# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18490 AC1-AC4: ticked boxes name an open ticket, never prove Done."""

from __future__ import annotations

import json
import logging

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_handler_evidence_autoclose_sweep import (
    FakeLinearClient,
    _dod_verify_ok,
    _issue,
    _make_gh_fake,
    _merged_pr,
    _request,
)

pytestmark = pytest.mark.unit
_TICKET = "OMN-9999"
_CHECKED = "## Acceptance criteria\n- [x] AC1: alpha\n  * [X] AC2: beta\n"


def _handler(
    linear: FakeLinearClient, *, green: bool = False
) -> HandlerEvidenceAutocloseSweep:
    async def verify(ticket: str, cwd: str, timeout: float):
        return (
            _dod_verify_ok(total=1, verified=int(green), failed=int(not green)),
            int(not green),
            "",
        )

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,
        run_gh_command=_make_gh_fake(
            [_merged_pr(100, f"evidence({_TICKET}): companion", _TICKET)],
            {100: [f"contracts/{_TICKET}.yaml"]},
        ),
        run_dod_verify_command=verify,
    )


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("green", [False, True])
async def test_started_ticked_ticket_is_named_in_one_tick_without_a_flip(
    apply, green, caplog
):
    linear = FakeLinearClient(issues={_TICKET: _issue(description=_CHECKED)})
    with caplog.at_level(logging.INFO):
        result = await _handler(linear, green=green).handle(_request(apply=apply))
    outcome = result.outcomes[0]

    assert outcome.decision.value == "gap_ticked_open"
    assert outcome.ticket_id == _TICKET
    assert result.tickets_flipped == 0
    assert linear.state_updates == []
    assert len(linear.comments) == int(apply)
    assert result.tickets_gap_posted == 1
    receipt = json.loads(result.model_dump_json())
    assert receipt["tickets_ticked_open"] == 1
    assert receipt["outcomes"][0]["decision"] == "gap_ticked_open"
    assert receipt["outcomes"][0]["ticket_id"] == _TICKET
    assert f"gap_ticked_open {_TICKET}" in caplog.text
    if apply:
        assert "author's assertion" in linear.comments[0][1]
        assert _TICKET in linear.comments[0][1]


@pytest.mark.parametrize(
    "description",
    [
        "## Acceptance criteria\n- [x] AC1: alpha\n- [ ] AC2: beta\n",
        "## Acceptance criteria\n- AC1: alpha\n",
        None,
    ],
)
async def test_unticked_or_absent_checkboxes_do_not_emit_the_decision(description):
    linear = FakeLinearClient(issues={_TICKET: _issue(description=description)})
    result = await _handler(linear).handle(_request(apply=True))
    assert result.outcomes[0].decision.value == "gap_posted"
    assert linear.state_updates == []


async def test_ticked_open_comment_is_posted_once_across_ticks():
    linear = FakeLinearClient(issues={_TICKET: _issue(description=_CHECKED)})
    handler = _handler(linear)
    first = await handler.handle(_request(apply=True))
    second = await handler.handle(_request(apply=True))
    assert first.outcomes[0].decision.value == "gap_ticked_open"
    assert second.outcomes[0].decision.value == "skipped_duplicate_comment"
    assert len(linear.comments) == 1
    assert linear.state_updates == []


@pytest.mark.parametrize("state", ["backlog", "unstarted", "completed", "canceled"])
async def test_only_started_candidates_emit_the_decision(state):
    linear = FakeLinearClient(
        issues={_TICKET: _issue(state_type=state, description=_CHECKED)}
    )
    result = await _handler(linear).handle(_request(apply=True))
    assert result.outcomes[0].decision.value != "gap_ticked_open"
    assert linear.state_updates == []


async def test_caller_exclusion_still_refuses_before_the_classification():
    linear = FakeLinearClient(issues={_TICKET: _issue(description=_CHECKED)})
    result = await _handler(linear).handle(
        _request(apply=True, exclude_tickets=(_TICKET,))
    )
    assert result.outcomes[0].decision.value == "skipped_excluded"
    assert linear.comments == []
    assert linear.state_updates == []


async def test_open_children_still_hold_the_grouped_candidate():
    issue = _issue(description=_CHECKED)
    issue["children"] = {
        "nodes": [{"identifier": "OMN-10000", "state": {"type": "started"}}]
    }
    linear = FakeLinearClient(issues={_TICKET: issue})
    result = await _handler(linear).handle(_request(apply=True))
    assert result.outcomes[0].decision.value == "skipped_has_children"
    assert linear.comments == []
    assert linear.state_updates == []


async def test_unreadable_comment_history_still_fails_closed():
    class UnreadableComments(FakeLinearClient):
        async def fetch_comment_bodies(self, issue_id: str) -> None:
            return None

    linear = UnreadableComments(issues={_TICKET: _issue(description=_CHECKED)})
    result = await _handler(linear).handle(_request(apply=True))
    assert result.outcomes[0].decision.value == "error_linear_api"
    assert linear.comments == []
    assert linear.state_updates == []


async def test_unarmed_schedule_names_the_ticket_without_writes():
    linear = FakeLinearClient(issues={_TICKET: _issue(description=_CHECKED)})
    result = await _handler(linear).handle(
        _request(trigger="schedule", scheduled_apply=False)
    )
    assert result.outcomes[0].decision.value == "gap_ticked_open"
    assert result.tickets_ticked_open == 1
    assert result.dry_run is True
    assert linear.comments == []
    assert linear.state_updates == []
