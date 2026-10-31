# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21731 — the standing arm enumerates tickets by contract, not by merge window.

The forward window (6 h) and the backfill window (168 h) are windows over merged
PRs, so a ticket whose PR merged longer ago was never offered: 598 of the 722
done-not-closed tickets in the 2026-10-31 triage. The ticket-contract arm lists
the key's own open tickets, intersects them with the ``contracts/<ticket>.yaml``
files on the default branch of the clones the run holds, and offers a bounded
rotating slice through the offer resolution.

Real git repositories in ``tmp_path`` carry the contracts and the squash
commits, so a resolver that reads the wrong ref or the wrong file list fails
here and not live. Groups: the aged ticket is offered with the windows
unchanged (AC1); every refusal and the width bound (AC2, ``refuses`` tests);
the run budget (AC3, ``budget`` test).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect import (
    offer_companion_sources as sources,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
    handler_evidence_autoclose_sweep as sweep_mod,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_arm import (
    EnumEvidenceAutocloseArm,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_16106_offer_selection import (
    _all_green,
    _RecordingLinear,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_21146_offer_local_companion import (
    _land,
    _make_clone,
)

pytestmark = pytest.mark.unit

_OWNER = "OmniNode-ai"
_OCC_REPO = f"{_OWNER}/onex_change_control"
_AGED = "OMN-31001"
_OLD = datetime.now(tz=UTC) - timedelta(days=30)
_RECENT = datetime.now(tz=UTC) - timedelta(hours=30)


class _CensusLinear(_RecordingLinear):
    """A recording Linear whose open-ticket census is scripted."""

    def __init__(
        self, open_ids: frozenset[str] | None, error: str = "census down"
    ) -> None:
        super().__init__()
        self._open_ids = open_ids
        self._error = error
        self.census_reads = 0

    async def fetch_open_ticket_ids(self) -> tuple[frozenset[str] | None, str]:
        self.census_reads += 1
        if self._open_ids is None:
            return None, self._error
        return self._open_ids, ""


async def _no_window_prs(args: list[str], timeout: float) -> tuple[Any, str]:
    """`gh api` for the window arms: nothing merged in either window."""
    path = args[2]
    if path.startswith("search/issues"):
        raise AssertionError("GitHub search was called: the guard refuses it")
    if "/pulls?" in path:
        return [], ""
    raise AssertionError(f"unexpected gh api path: {path}")


def _handler(
    linear: _CensusLinear,
) -> tuple[HandlerEvidenceAutocloseSweep, list[str]]:
    verified: list[str] = []

    async def fake_dod_verify(
        ticket_id: str, cwd: str, timeout: float
    ) -> tuple[dict[str, Any], int, str]:
        verified.append(ticket_id)
        return _all_green(ticket_id), 0, ""

    return (
        HandlerEvidenceAutocloseSweep(
            linear_client=cast("Any", linear),
            run_gh_command=_no_window_prs,
            run_dod_verify_command=fake_dod_verify,
        ),
        verified,
    )


def _request(root: Path, **overrides: Any) -> ModelEvidenceAutocloseSweepRequest:
    defaults: dict[str, Any] = {
        "correlation_id": uuid4(),
        "evidence_repo": _OCC_REPO,
        "lookback_hours": 6,
        "backfill_lookback_hours": 168,
        "apply": True,
        "clone_root": str(root),
        "pr_watcher_state_path": str(root / "no-watcher.json"),
        "demotion_window_hours": 0,
    }
    defaults.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**defaults)


@pytest.fixture
def clones(tmp_path: Path) -> Path:
    root = tmp_path / "clones"
    root.mkdir()
    return root


def _land_ticket(
    root: Path, repo: str, ticket: str, number: int, when: datetime
) -> None:
    clone = root / repo
    if not clone.exists():
        _make_clone(root, repo)
    _land(
        clone,
        f"feat({ticket}): the work (#{number})",
        [f"contracts/{ticket}.yaml"],
        when=when,
    )


# =====================================================================
# AC1 — the aged ticket is offered, the windows unchanged
# =====================================================================


async def test_a_ticket_merged_beyond_the_backfill_window_is_offered(
    clones: Path,
) -> None:
    _land_ticket(clones, "omnimarket", _AGED, 4401, _OLD)
    linear = _CensusLinear(frozenset({_AGED}))
    handler, verified = _handler(linear)

    request = _request(clones)
    assert request.lookback_hours == 6
    assert request.backfill_lookback_hours == 168
    result = await handler.handle(request)

    contract_outcomes = [
        o
        for o in result.outcomes
        if o.enumeration_arm is EnumEvidenceAutocloseArm.CONTRACT
    ]
    assert [o.ticket_id for o in contract_outcomes] == [_AGED]
    assert contract_outcomes[0].companion_pr_number == 4401
    assert result.contract_pool_size == 1
    assert result.contract_candidates_selected == 1
    assert result.companions_scanned == 1
    assert linear.census_reads == 1
    assert _AGED in verified, "the offered ticket went through the verifier"


async def test_the_default_request_arms_the_arm(clones: Path) -> None:
    assert (
        ModelEvidenceAutocloseSweepRequest.model_fields[
            "contract_enumeration_max_candidates"
        ].default
        == 10
    )
    assert _request(clones).contract_enumeration_max_candidates == 10


async def test_a_ticket_with_the_same_pr_as_another_is_not_guessed(
    clones: Path,
) -> None:
    """One squash PR whose contracts bind two open tickets is dropped for both."""
    _make_clone(clones, "omnimarket")
    _land(
        clones / "omnimarket",
        "feat(OMN-31010, OMN-31011): two tickets (#4410)",
        ["contracts/OMN-31010.yaml", "contracts/OMN-31011.yaml"],
        when=_OLD,
    )
    linear = _CensusLinear(frozenset({"OMN-31010", "OMN-31011"}))
    handler, verified = _handler(linear)

    result = await handler.handle(_request(clones))

    assert result.contract_candidates_selected == 0
    assert verified == []


# =====================================================================
# AC2 — what the arm refuses, and the width bound
# =====================================================================


async def test_the_arm_refuses_a_ticket_the_census_does_not_list(
    clones: Path,
) -> None:
    """Done, canceled and a collaborator's ticket are all 'not in the census'."""
    _land_ticket(clones, "omnimarket", _AGED, 4401, _OLD)
    linear = _CensusLinear(frozenset({"OMN-99999"}))
    handler, verified = _handler(linear)

    result = await handler.handle(_request(clones))

    assert result.contract_pool_size == 0
    assert result.contract_candidates_selected == 0
    assert verified == []
    assert _AGED not in linear.reads


async def test_the_arm_refuses_a_ticket_on_the_exclusion_fence(clones: Path) -> None:
    _land_ticket(clones, "omnimarket", _AGED, 4401, _OLD)
    linear = _CensusLinear(frozenset({_AGED}))
    handler, verified = _handler(linear)

    result = await handler.handle(_request(clones, exclude_tickets=(_AGED,)))

    assert result.contract_candidates_selected == 0
    assert verified == []
    assert _AGED not in linear.reads


async def test_the_arm_refuses_an_open_contract_with_no_merged_pr_and_says_so(
    clones: Path,
) -> None:
    clone = _make_clone(clones, "omnimarket")
    _land(
        clone,
        "docs: a contract with no ticket in the subject",
        [f"contracts/{_AGED}.yaml"],
    )
    # The commit names no PR number, so there is no merged PR to bind.
    linear = _CensusLinear(frozenset({_AGED}))
    handler, verified = _handler(linear)

    result = await handler.handle(_request(clones))

    mine = [
        o
        for o in result.outcomes
        if o.enumeration_arm is EnumEvidenceAutocloseArm.CONTRACT
    ]
    assert [o.decision for o in mine] == [
        EnumEvidenceAutocloseDecision.SKIPPED_NO_OFFER_COMPANION
    ]
    assert mine[0].ticket_id == _AGED
    assert result.contract_candidates_selected == 0
    assert verified == []


async def test_the_arm_refuses_a_ticket_inside_the_backfill_window(
    clones: Path,
) -> None:
    """That ticket belongs to the backfill arm's rotation, not this one."""
    _land_ticket(clones, "omnimarket", _AGED, 4402, _RECENT)
    linear = _CensusLinear(frozenset({_AGED}))
    handler, _verified = _handler(linear)

    result = await handler.handle(_request(clones))

    assert result.contract_candidates_selected == 0
    assert not [
        o
        for o in result.outcomes
        if o.enumeration_arm is EnumEvidenceAutocloseArm.CONTRACT
    ]


async def test_an_unreadable_census_refuses_the_arm_and_leaves_the_other_arms_running(
    clones: Path,
) -> None:
    _land_ticket(clones, "omnimarket", _AGED, 4401, _OLD)
    linear = _CensusLinear(None, error="HTTP 503")
    handler, verified = _handler(linear)

    result = await handler.handle(_request(clones))

    assert result.success is True
    assert result.contract_candidates_selected == 0
    errors = [
        o
        for o in result.outcomes
        if o.enumeration_arm is EnumEvidenceAutocloseArm.CONTRACT
    ]
    assert [o.decision for o in errors] == [
        EnumEvidenceAutocloseDecision.ERROR_LINEAR_API
    ]
    assert "HTTP 503" in errors[0].reason
    assert verified == []


async def test_the_slice_is_never_wider_than_the_contract_field(clones: Path) -> None:
    tickets = [f"OMN-3110{n}" for n in range(6)]
    for index, ticket in enumerate(tickets):
        _land_ticket(clones, "omnimarket", ticket, 4500 + index, _OLD)
    linear = _CensusLinear(frozenset(tickets))
    handler, _verified = _handler(linear)

    result = await handler.handle(
        _request(clones, contract_enumeration_max_candidates=2)
    )

    assert result.contract_pool_size == 6
    assert result.contract_candidates_selected == 2


async def test_a_width_of_zero_turns_the_arm_off_without_a_linear_read(
    clones: Path,
) -> None:
    _land_ticket(clones, "omnimarket", _AGED, 4401, _OLD)
    linear = _CensusLinear(frozenset({_AGED}))
    handler, verified = _handler(linear)

    result = await handler.handle(
        _request(clones, contract_enumeration_max_candidates=0)
    )

    assert linear.census_reads == 0
    assert result.contract_pool_size == 0
    assert verified == []


def test_the_rotation_covers_every_ticket_of_the_pool() -> None:
    pool = [f"OMN-{n}" for n in range(1, 24)]
    covered: set[str] = set()
    for tick in range(3):
        covered |= set(sweep_mod._rotating_slice(pool, 10, tick))
    assert covered == set(pool)


def test_list_contract_tickets_reads_every_clone_but_change_control(
    clones: Path,
) -> None:
    _land_ticket(clones, "omnimarket", "OMN-31020", 1, _OLD)
    _land_ticket(clones, "omnibase_infra", "OMN-31021", 2, _OLD)
    _land_ticket(clones, "onex_change_control", "OMN-31022", 3, _OLD)

    found = sources.list_contract_tickets(
        clones, "onex_change_control", sources.run_git_real
    )

    assert found == {"OMN-31020", "OMN-31021"}
    assert sources.list_contract_tickets(None, "x", sources.run_git_real) == set()


# =====================================================================
# the Linear census: scoped to the key's own open tickets, fails closed
# =====================================================================


def test_the_census_query_is_scoped_to_the_keys_own_non_terminal_tickets() -> None:
    query = sweep_mod._OPEN_TICKETS_QUERY
    assert "isMe: { eq: true }" in query, "a collaborator's ticket must not be listed"
    assert '"completed"' in query
    assert '"canceled"' in query


class _PagedClient(sweep_mod._LinearClient):
    def __init__(self, pages: list[dict[str, Any] | None]) -> None:
        super().__init__(api_key="k")
        self._pages = list(pages)
        self.cursors: list[object] = []

    async def _query(
        self, query: str, variables: dict[str, Any]
    ) -> dict[str, Any] | None:
        self.cursors.append(variables.get("after"))
        return self._pages.pop(0)


def _page(ids: list[str], *, next_cursor: str | None) -> dict[str, Any]:
    return {
        "issues": {
            "pageInfo": {
                "hasNextPage": next_cursor is not None,
                "endCursor": next_cursor,
            },
            "nodes": [{"identifier": i} for i in ids],
        }
    }


async def test_the_census_follows_cursors_and_returns_every_page() -> None:
    client = _PagedClient(
        [
            _page(["OMN-1", "omn-2"], next_cursor="c1"),
            _page(["OMN-3"], next_cursor=None),
        ]
    )
    found, error = await client.fetch_open_ticket_ids()
    assert error == ""
    assert found == frozenset({"OMN-1", "OMN-2", "OMN-3"})
    assert client.cursors == [None, "c1"]


async def test_the_census_refuses_a_partial_read() -> None:
    failed_midway = _PagedClient([_page(["OMN-1"], next_cursor="c1"), None])
    found, error = await failed_midway.fetch_open_ticket_ids()
    assert found is None and "Failed to fetch open tickets" in error

    no_cursor = _PagedClient(
        [{"issues": {"pageInfo": {"hasNextPage": True}, "nodes": []}}]
    )
    found, error = await no_cursor.fetch_open_ticket_ids()
    assert found is None and "no cursor" in error

    malformed = _PagedClient([{"issues": {"nodes": "x"}}])
    found, error = await malformed.fetch_open_ticket_ids()
    assert found is None and "Malformed" in error


# =====================================================================
# AC3 — the run budget
# =====================================================================

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"
# The 8 scheduled runs of 2026-10-30T20:10Z to 2026-10-31T10:09Z: the slowest
# sweep step was 285 s and reached the verifier for 9 candidates (31.7 s each,
# the run's own enumeration included). Every one of them selected 5 backfill
# candidates.
_WORST_SWEEP_SECONDS = 285.0
_WORST_CALLS = 9
_MEASURED_BACKFILL_WIDTH = 5


def test_the_run_budget_with_both_arms_at_their_default_widths_fits_the_timeout() -> (
    None
):
    fields = ModelEvidenceAutocloseSweepRequest.model_fields
    backfill = fields["backfill_max_candidates"].default
    contract = fields["contract_enumeration_max_candidates"].default
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    timeout = min(
        float(job["timeout-minutes"]) * 60
        for job in workflow["jobs"].values()
        if "timeout-minutes" in job
    )
    per_call = _WORST_SWEEP_SECONDS / _WORST_CALLS
    projected = (
        _WORST_SWEEP_SECONDS
        + (max(backfill - _MEASURED_BACKFILL_WIDTH, 0) + contract) * per_call
    )
    assert projected <= timeout * 0.5, (
        f"backfill {backfill} + contract {contract} project a {projected:.0f}s "
        f"sweep at {per_call:.1f}s per call, over half of the {timeout:.0f}s job "
        "timeout"
    )
