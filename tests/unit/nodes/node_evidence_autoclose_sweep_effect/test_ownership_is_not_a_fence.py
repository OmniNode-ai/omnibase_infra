# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Closure is ownership-agnostic; only a concurrent write may hold a ticket.

Operator ruling, firm, 2026-09-05T20:45:51Z, recorded at
`docs/tracking/ROLLING_WORK_LEDGER.md:3372`: when the acceptance criteria are
met on live evidence the ticket is closed, whoever it is assigned to. Assignee
and fence ownership never hold a Done-eligible ticket open.

The OMN-17891 fence was being used against that. It had accumulated a static
list of 26 ticket ids on the `ONEX_AUTOCLOSE_EXCLUDE` repo variable, derived
from who was working them rather than from any concurrent write — 26 tickets
the closer refused to look at, indefinitely, and silently: `SKIPPED_EXCLUDED`
is recorded before any Linear read, so a fenced ticket produces no verdict at
all and nothing on the board says why.

The mechanism stays; what it may assert is narrowed to the one thing the
sweep genuinely cannot derive for itself — a live ledger CLAIM row saying
another lane is writing this ticket right now. These tests pin that the node
carries no ownership knowledge of its own: no ticket id is baked into any
default, and the only refusals it can reach are caller-supplied, label,
already-done, and the children fence.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
    handler_evidence_autoclose_sweep as sweep_mod,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models import (
    model_evidence_autoclose_sweep_request as request_mod,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

from .test_handler_evidence_autoclose_sweep import (
    FakeLinearClient,
    _issue,
    _make_dod_verify_fake,
    _make_gh_fake,
    _merged_pr,
    _request,
)
from .test_live_check_not_executed_hold import _check, _receipt

pytestmark = pytest.mark.unit

#: One of the 26 ids that sat on the standing variable on 2026-09-05.
_FORMERLY_FENCED = "OMN-17857"

_TICKET_LITERAL = re.compile(r"OMN-\d+")


def _met_dod() -> dict[str, object]:
    return _receipt(
        checks=[
            # OMN-18056: the behaviour proof declares WHICH criterion it
            # covers -- the one the shared `_issue` body labels `AC1`.
            _check(
                "dod-behaviour",
                "verified",
                proof_class="behavior",
                binds_ac=("AC1",),
            ),
            _check("dod-merge", "verified", proof_class="merge-state"),
        ],
        verdict_status="verified",
        behavior_proving=1,
    )


async def _sweep(
    *, ticket: str, exclude: tuple[str, ...], ticks: int = 1
) -> tuple[object, FakeLinearClient]:
    """Run the sweep ``ticks`` times against one fake board, returning the last.

    OMN-18056: a flip takes two ticks -- the first eligible observation of a
    verdict arms a re-draw and writes no Done, and the second one on the same
    fingerprint closes. A refusal that precedes the re-draw needs only one.
    """
    linear = FakeLinearClient(issues={ticket: _issue()})
    handler = HandlerEvidenceAutocloseSweep(
        linear_client=linear,
        run_gh_command=_make_gh_fake(
            companions=[_merged_pr(1, f"evidence({ticket}): x", ticket)],
            files_by_pr={1: [f"contracts/{ticket}.yaml"]},
        ),
        run_dod_verify_command=_make_dod_verify_fake({ticket: (_met_dod(), 0, "")}),
    )
    result = None
    for _tick in range(ticks):
        result = await handler.handle(_request(apply=True, exclude_tickets=exclude))
    assert result is not None
    return result.outcomes[0], linear


async def test_a_formerly_fenced_ticket_with_met_dod_flips() -> None:
    """The ruling, executable. Ownership is not a reason to withhold a close."""
    outcome, linear = await _sweep(ticket=_FORMERLY_FENCED, exclude=(), ticks=2)

    assert outcome.decision == EnumEvidenceAutocloseDecision.FLIPPED
    assert len(linear.state_updates) == 1


async def test_a_ticket_a_concurrent_lane_is_writing_is_still_refused() -> None:
    """The one authority that survives: a live CLAIM row, caller-supplied."""
    outcome, linear = await _sweep(ticket=_FORMERLY_FENCED, exclude=(_FORMERLY_FENCED,))

    assert outcome.decision == EnumEvidenceAutocloseDecision.SKIPPED_EXCLUDED
    assert linear.state_updates == []
    assert linear.fetch_issue_calls == []


def test_the_fence_has_no_standing_membership_of_its_own() -> None:
    """No ticket is fenced unless this run's caller says so."""
    assert (
        ModelEvidenceAutocloseSweepRequest.model_fields["exclude_tickets"].default == ()
    )


def test_no_ticket_id_is_baked_into_the_node_source() -> None:
    """A ticket id in this node's code is either a citation or a fence.

    Citations live in comments and docstrings; a fence would live in an
    expression. This reads the source with comments and string literals
    removed, so a hard-coded refusal set cannot hide behind prose.
    """
    for module in (sweep_mod, request_mod):
        source = inspect.getsource(module)
        code_only = _strip_comments_and_strings(source)
        assert not _TICKET_LITERAL.search(code_only), (
            f"{module.__name__} names a ticket id in executable code — "
            "the fence is caller-supplied (ledger:3372)"
        )


def test_the_fence_field_names_the_authority_it_accepts() -> None:
    """The field description is what the next dispatcher reads before setting it."""
    description = (
        ModelEvidenceAutocloseSweepRequest.model_fields["exclude_tickets"].description
        or ""
    )
    assert "CLAIM" in description
    assert "ownership" in description.lower()
    assert "3372" in description


def test_the_workflow_carries_no_standing_ticket_list() -> None:
    """The variable is read; its content is never spelled in the repo.

    A list committed here would be exactly the static ownership fence the
    ruling forbids, re-introduced where no ledger row can contradict it.
    """
    workflow = (
        Path(__file__).resolve().parents[4]
        / ".github"
        / "workflows"
        / "evidence-autoclose-sweep.yml"
    )
    assignments = [
        line.strip()
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if re.match(r"^(STANDING_)?EXCLUDE_TICKETS:", line.strip())
    ]
    assert len(assignments) == 2, (
        "expected exactly the standing and the dispatch fence assignments, "
        f"got {assignments}"
    )
    for assignment in assignments:
        assert not _TICKET_LITERAL.search(assignment), (
            f"a ticket id spelled into a fence assignment is a standing "
            f"ownership fence, committed where no ledger row can contradict "
            f"it: {assignment}"
        )
        assert "vars." in assignment or "github.event.inputs." in assignment, (
            f"a fence must resolve from a variable or a dispatch input, not "
            f"from a constant: {assignment}"
        )


def _strip_comments_and_strings(source: str) -> str:
    """Remove `#` comments and every string literal from Python source."""
    without_strings = re.sub(
        r'("""|\'\'\')(?:.|\n)*?\1', '""', source, flags=re.MULTILINE
    )
    without_strings = re.sub(r'"[^"\n]*"', '""', without_strings)
    without_strings = re.sub(r"'[^'\n]*'", "''", without_strings)
    return "\n".join(line.split("#", 1)[0] for line in without_strings.splitlines())
