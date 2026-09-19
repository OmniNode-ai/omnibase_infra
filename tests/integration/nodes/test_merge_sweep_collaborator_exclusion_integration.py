# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The collaborator exclusion holds across the whole merge-sweep chain (OMN-18823).

The unit tests hand `_classify_single` a `ModelPRInfo` that already carries
assignees. That proves the rule and proves nothing about the wiring: the rule
is worthless if the scan never asks GitHub for the field, or if the
orchestrator drops the roster on its way to the classifier. Both of those are
the shape of the original defect — every layer was individually reasonable and
the fact never travelled.

So this walks the real chain: the PR-list EFFECT over a gh payload (the
subprocess is the only thing faked), into the ORCHESTRATOR handler that builds
the classify input, into the COMPUTE classifier. Occurrence OMN-18794,
omniweb#423, squash-merged 2026-09-19T12:33:45Z.

Every login here is synthetic. omnibase_infra is public and the real roster
lives in the private vocabulary home the node contract names.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_merge_sweep_classify_compute.handlers.handler_classify_prs import (
    HandlerClassifyPRs,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.enum_classify_skip_reason import (
    EnumClassifySkipReason,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.handlers.handler_pr_list import (
    HandlerPRList,
)
from omnibase_infra.nodes.node_merge_sweep_workflow_orchestrator.handlers.handler_pr_list_complete import (
    HandlerPRListComplete,
)

pytestmark = pytest.mark.integration

_ROSTER = ("collab-one", "collab-two")

_GREEN_ROLLUP = [{"conclusion": "SUCCESS", "state": "SUCCESS"}]


def _gh_pr(number: int, **overrides: object) -> dict[str, object]:
    """A gh `pr list --json` object for a PR that would otherwise be Track A."""
    payload: dict[str, object] = {
        "number": number,
        "title": f"PR {number}",
        "headRefName": f"branch-{number}",
        "baseRefName": "dev",
        "author": {"login": "author-login"},
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "reviewDecision": "APPROVED",
        "statusCheckRollup": _GREEN_ROLLUP,
        "autoMergeRequest": None,
        "labels": [],
        "updatedAt": "2026-09-19T12:00:00Z",
        "assignees": [],
        "reviewRequests": [],
    }
    payload.update(overrides)
    return payload


async def _scan(prs: list[dict[str, object]]):
    """Run the real PR-list effect over a faked gh subprocess."""
    proc = AsyncMock()
    proc.communicate.return_value = (json.dumps(prs).encode(), b"")
    proc.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        return await HandlerPRList().handle(
            repos=("OmniNode-ai/test",), correlation_id=uuid4()
        )


async def _chain(prs: list[dict[str, object]], roster: tuple[str, ...] = _ROSTER):
    """EFFECT -> ORCHESTRATOR -> COMPUTE, as the workflow wires them."""
    scanned = await _scan(prs)
    classify_input = await HandlerPRListComplete().handle(
        correlation_id=scanned.correlation_id,
        prs=scanned.prs,
        collaborator_logins=roster,
    )
    return await HandlerClassifyPRs().handle(classify_input)


@pytest.mark.asyncio
async def test_the_scan_carries_the_hand_off_fields_through_to_the_model() -> None:
    """The effect must actually observe both fields, or the rule cannot fire."""
    scanned = await _scan(
        [
            _gh_pr(
                1,
                assignees=[{"login": "collab-one"}],
                reviewRequests=[{"login": "collab-two"}, {"slug": "platform-leads"}],
            )
        ]
    )

    (pr,) = scanned.prs
    assert pr.assignees == ("collab-one",)
    assert pr.requested_reviewers == ("collab-two", "platform-leads")


@pytest.mark.asyncio
async def test_a_pr_assigned_to_a_collaborator_is_withheld_end_to_end() -> None:
    """The exact shape of omniweb#423: green, approved, mergeable, assigned."""
    result = await _chain([_gh_pr(423, assignees=[{"login": "collab-one"}])])

    assert len(result.track_a) == 0, "a PR handed to a person must not be admitted"
    assert len(result.track_b) == 0, "nor sent to the branch-mutating polish track"
    (skipped,) = result.skipped
    assert skipped.skip_reason is EnumClassifySkipReason.COLLABORATOR_EXCLUDED
    assert skipped.excluded_account == "collab-one"
    assert skipped.pr.number == 423


@pytest.mark.asyncio
async def test_a_requested_reviewer_alone_is_enough_to_withhold() -> None:
    """Nobody assigned, review requested: still a hand-off."""
    result = await _chain([_gh_pr(424, reviewRequests=[{"login": "collab-two"}])])

    assert len(result.track_a) == 0
    assert result.skipped[0].excluded_account == "collab-two"


@pytest.mark.asyncio
async def test_an_unassigned_pr_still_reaches_track_a() -> None:
    """Positive control: the chain still admits what it always admitted.

    Without this, a chain that skipped everything would pass the two tests
    above and read as a working exclusion.
    """
    result = await _chain([_gh_pr(425)])

    assert len(result.skipped) == 0
    assert len(result.track_a) == 1
    assert result.track_a[0].pr.number == 425


@pytest.mark.asyncio
async def test_one_batch_separates_the_hand_offs_from_the_rest() -> None:
    """A realistic sweep: two admissible, two withheld, one unrelated assignee."""
    result = await _chain(
        [
            _gh_pr(1),
            _gh_pr(2, assignees=[{"login": "collab-one"}]),
            _gh_pr(3, reviewRequests=[{"login": "collab-two"}]),
            _gh_pr(4, assignees=[{"login": "someone-else"}]),
        ]
    )

    assert result.total_classified == 4
    assert {c.pr.number for c in result.track_a} == {1, 4}
    assert {c.excluded_account for c in result.skipped} == {"collab-one", "collab-two"}


@pytest.mark.asyncio
async def test_an_empty_roster_admits_the_same_pull_request() -> None:
    """The roster, not the field, is what withholds — and it is caller-declared."""
    result = await _chain([_gh_pr(426, assignees=[{"login": "collab-one"}])], roster=())

    assert len(result.skipped) == 0
    assert len(result.track_a) == 1
