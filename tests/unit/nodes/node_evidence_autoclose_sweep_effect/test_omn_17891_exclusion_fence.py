# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17891 — a caller-assertable per-candidate fence for the apply path.

The sweep's only refusals were a Linear label, an already-completed state, a
binding-hygiene skip, and the GLOBAL ``ONEX_AUTOCLOSE_DISABLED`` kill switch.
None of those can refuse ONE named candidate: the label has to be set in Linear
beforehand, and the kill switch aborts the entire run with zero I/O.

Measured consequence (dry run 33859059657, 2026-09-04T09:35Z): 25 companions
scanned, 7 tickets would flip, and 6 of those 7 were unsafe for reasons living
entirely outside the node's reach — an open ledger CLAIM, a RED staging boot
gate, a ``HOLD`` comment, or ownership by a concurrent controller. The first
applied run had no way to flip the one adjudicated-safe ticket without exposing
the other six to the same run, because the workflow's only knobs are
``lookback_hours`` (a companion-merge window, not a ticket selector) and
``apply``.

So the request carries ``exclude_tickets`` and the sweep records
``SKIPPED_EXCLUDED``. Two properties are load-bearing and both are asserted
below:

1. **The refusal happens BEFORE the first Linear read.** An excluded ticket
   costs zero Linear I/O, so no API failure, no verdict, and no later decision
   branch can convert an exclusion into something else — in particular it can
   never be reported as ``ERROR_LINEAR_API``, which reads as "the fence did not
   apply" rather than "the fence applied".
2. **It is caller-asserted, never derived.** The node reads no ledger and no
   ownership signal; the value is supplied by whoever dispatches the run. That
   is why the decision is a distinct enum value rather than a reuse of
   ``SKIPPED_LABEL``: the audit record must say which authority refused.

The kill switch is unaffected and is re-asserted here — an exclusion list is a
per-dispatch choice, and it must not become a way to opt one ticket back INTO a
halted run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_TICKET = "OMN-17857"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)


def _merged_pr(number: int) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({_TICKET}): OCC companion",
        "updated_at": recent,
        "merged_at": recent,
    }


def _flip_clearing_skill_result() -> dict[str, object]:
    """A verdict that clears the flip predicate outright.

    Two verified checks, one of them behaviour-proving, zero failed, zero
    non-probative, and a description with no acceptance-criteria checkboxes —
    so nothing except the fence can be what withholds the flip. Without the
    fence this candidate flips, which is the point.
    """
    terminal: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": "verified",
        "dry_run": False,
        "checks": [
            {
                "evidence_id": "dod-pr-state",
                "description": "dod-pr-state",
                "status": "verified",
                "message": "OK (1ms)",
                "proof_class": "merge-state",
            },
            {
                "evidence_id": "dod-tests",
                "description": "dod-tests",
                "status": "verified",
                "message": "OK (1ms)",
                "proof_class": "behavior",
            },
        ],
        "total_checks": 2,
        "verified_count": 2,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "behavior_proving_count": 1,
        "error_message": None,
    }
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "exit_code": 0,
        "result": terminal,
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


class _RecordingLinear:
    """Records every call, reads included.

    ``reads`` is what proves property 1: an excluded candidate must not appear
    in it at all. A double that only logged writes could not tell "refused
    before the read" from "read, then refused".
    """

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object]:
        self.reads.append(ticket_id)
        return {
            "id": "issue-uuid-1",
            "identifier": ticket_id,
            "state": {"id": "s1", "name": "In Progress", "type": "started"},
            "labels": {"nodes": []},
            "team": {"id": "team-1"},
            "description": None,
        }

    async def fetch_done_state_id(self, team_id: str) -> str:
        return "state-done-id"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)


def _handler(
    linear: _RecordingLinear, *, autoclose_disabled: bool = False
) -> HandlerEvidenceAutocloseSweep:
    async def fake_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            return [{"filename": f"contracts/{_TICKET}.yaml"}], ""
        page = int(path.rsplit("page=", 1)[1])
        return ([_merged_pr(8174)], "") if page == 1 else ([], "")

    async def fake_dod_verify(ticket_id: str, cwd: str, timeout: float):
        return _flip_clearing_skill_result(), 0, ""

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=autoclose_disabled,
        run_gh_command=fake_gh,
        run_dod_verify_command=fake_dod_verify,
    )


def _request(**overrides: object) -> ModelEvidenceAutocloseSweepRequest:
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 6,
        "apply": True,
    }
    defaults.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**defaults)


async def test_an_excluded_candidate_is_refused_under_apply_and_never_mutated() -> None:
    """The blocker, executed: a would-flip candidate on the exclusion list.

    The verdict clears the predicate, ``apply=True``, and the ticket is still
    not flipped, not commented, and not even READ.
    """
    linear = _RecordingLinear()
    handler = _handler(linear)

    result = await handler.handle(_request(exclude_tickets=(_TICKET,)))

    assert result.tickets_flipped == 0
    assert result.tickets_skipped == 1
    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.SKIPPED_EXCLUDED
    assert outcome.ticket_id == _TICKET
    assert outcome.applied is False
    assert outcome.linear_comment_posted is False
    # Property 1: refused before ANY Linear call, reads included.
    assert linear.reads == []
    assert linear.state_updates == []
    assert linear.comments == []


async def test_the_same_candidate_flips_when_the_list_is_empty() -> None:
    """The positive control.

    Without it, a test asserting "not flipped" proves nothing — the candidate
    might have been withheld by the predicate, the behaviour-proof gate, or the
    AC-coverage guard rather than by the fence.
    """
    linear = _RecordingLinear()
    handler = _handler(linear)

    result = await handler.handle(_request())

    assert result.tickets_flipped == 1
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert linear.reads == [_TICKET]
    assert linear.state_updates == [("issue-uuid-1", "state-done-id")]


@pytest.mark.parametrize(
    "supplied",
    [
        ("omn-17857",),
        ("  OMN-17857  ",),
        ("OMN-99999", "OMN-17857"),
    ],
    ids=["lowercase", "surrounding-whitespace", "one-of-several"],
)
async def test_matching_ignores_case_and_surrounding_whitespace(
    supplied: tuple[str, ...],
) -> None:
    """A fence that misses on a typo is worse than no fence.

    The value is typed by an operator into a workflow_dispatch box or spliced
    from a script's output, so ``omn-17857`` and a trailing space are the
    expected shapes, not edge cases. A near-miss here does not fail loudly — it
    flips the ticket the operator was fencing off.
    """
    linear = _RecordingLinear()
    handler = _handler(linear)

    result = await handler.handle(_request(exclude_tickets=supplied))

    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.SKIPPED_EXCLUDED
    assert linear.reads == []
    assert linear.state_updates == []


async def test_a_non_matching_exclusion_list_does_not_fence_anything() -> None:
    """Only the named tickets are refused; the list is not a global halt."""
    linear = _RecordingLinear()
    handler = _handler(linear)

    result = await handler.handle(_request(exclude_tickets=("OMN-99999",)))

    assert result.tickets_flipped == 1
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert linear.state_updates == [("issue-uuid-1", "state-done-id")]


async def test_the_default_is_no_exclusions() -> None:
    """Byte-identical behaviour for every existing caller.

    The field is additive on a frozen, ``extra="forbid"`` model, so a caller
    that never heard of it must be unaffected.
    """
    request = ModelEvidenceAutocloseSweepRequest(correlation_id=uuid4())
    assert request.exclude_tickets == ()


async def test_the_kill_switch_still_dominates_an_exclusion_scoped_run() -> None:
    """An exclusion list must not become a way back INTO a halted run.

    The kill switch is global and does zero I/O; scoping a dispatch with
    ``exclude_tickets`` changes nothing about that.
    """
    linear = _RecordingLinear()
    handler = _handler(linear, autoclose_disabled=True)

    result = await handler.handle(_request(exclude_tickets=("OMN-99999",)))

    assert result.kill_switch_engaged is True
    assert result.companions_scanned == 0
    assert result.outcomes == ()
    assert linear.reads == []
    assert linear.state_updates == []
