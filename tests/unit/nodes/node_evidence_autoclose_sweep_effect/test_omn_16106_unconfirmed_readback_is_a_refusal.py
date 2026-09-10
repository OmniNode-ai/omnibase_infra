# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106 — the three predicate paths that closed OMN-16025 unearned.

Measured, not theorised. The scheduled sweep run
`34061364537 <https://github.com/OmniNode-ai/omnibase_infra/actions/runs/34061364537>`_
(started 2026-09-06T21:32:07Z, ``mode=apply_scheduled``) moved OMN-16025 —
"[Gate] Delegation canary green — five-link chain proven, the single
beta-is-real indicator" — from Backlog to Done at 21:42:17.523Z. Its outcome
row, verbatim from the job log:

    {"ticket_id":"OMN-16025","companion_pr_number":8436,
     "decision":"error_readback_unconfirmed",
     "dod_verify_total_checks":12,"dod_verify_verified_count":6,
     "dod_verify_failed_count":0,"dod_verify_non_probative_count":6,
     "dod_verify_behavior_proving_count":1,
     "uncovered_acceptance_criteria":[],
     "readback_entry_id":"","verdict_fingerprint":"9aa50d20ab385fd9",
     "applied":true}

The run reported ``tickets_flipped: 0`` while the board read Done. Three
independent defects, one test class each.

1. **An unconfirmed readback applied instead of refusing.** The Done stayed on
   the board under a comment that said "Treat this ticket's state as written
   but unverified, and check it by hand". The label and the flip cannot
   coexist: the board is what every downstream reader sees, the comment is
   what almost nobody opens.

2. **The OMN-16736 AC-coverage re-read saw no criteria at all.**
   ``uncovered_acceptance_criteria: []`` on a ticket whose body lists five
   numbered links and then says "Verified by golden-chain replay on the
   stability lane" — a lane that was deliberately never touched, stated in the
   lane lead's own comment six minutes before the flip. The parser only
   collected items under a recognised heading; this body has none, so both
   counting bounds sat behind an ``if not items`` early exit.

3. **Nothing consulted the gate's own probe.** ``chain-canary.yml`` was
   ``conclusion=failure`` on every run of 2026-09-06, including run
   34061981317 fired at 21:44:42Z — two minutes AFTER the flip — at 3 of 5
   links proven.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _ac_coverage_gap,
    _acceptance_criteria_items,
    _gate_probe_declaration,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect._ac_binding_support import (
    BOUND_AC_DESCRIPTION,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_TICKET = "OMN-16025"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)

# The live OMN-16025 description, byte-for-byte as Linear returned it on
# 2026-09-06. Note what it does NOT have: any markdown heading at all. The five
# criteria are a bare numbered list under a prose opener, and the lane
# requirement is a prose sentence outside the list.
_OMN_16025_DESCRIPTION = """Acceptance is the unified plan set 09 §5.1 chain. Each link must carry its own evidence line:

1. Intent submitted through the live gateway path.
2. Routing decision PUBLISHED and PROJECTED (readback from projection, not logs).
3. Delegated execution completes.
4. Emission OUTBOX-CONFIRMED via broker readback (not publish-return).
5. Complete ledger chain + replay green through an HONEST tier-2 verifier (SKIP≠PASS).

Verified by golden-chain replay on the stability lane. Demo success does not close this ticket.

This gate is blocked by the four defect fixes and must not flip until each is Done."""


def _companion(number: int, ticket: str) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({ticket}): OCC companion",
        "updated_at": recent,
        "merged_at": recent,
    }


def _receipt(
    *,
    total: int = 12,
    verified: int = 6,
    failed: int = 0,
    non_probative: int = 6,
    behavior: int = 1,
    binds_ac: tuple[str, ...] = ("AC1", "AC2"),
) -> dict[str, object]:
    """The OMN-16025 counters by default — 6 verified, 6 non-probative, 12.

    OMN-18056: one verified probative check DECLARES the criteria the bodies
    below label. The counters are read from the count fields and never from
    the checks list, so this binds the AC-binding gate without moving any
    arithmetic the gate-probe and readback conjuncts depend on. A fixture that
    means to model a corpus declaring NOTHING passes ``binds_ac=()``.
    """
    verdict: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": "verified",
        "dry_run": False,
        "checks": [
            {
                "evidence_id": "omn18056-bound-check",
                "status": "verified",
                "proof_class": "behavior",
                "binds_ac": list(binds_ac),
            }
        ],
        "total_checks": total,
        "verified_count": verified,
        "failed_count": failed,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": non_probative,
        "behavior_proving_count": behavior,
        "error_message": None,
    }
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 0,
        "duration_ms": 1,
        "result": verdict,
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


class FakeLinear:
    """A Linear whose history connection and rollback write are steerable."""

    def __init__(
        self,
        *,
        description: str = BOUND_AC_DESCRIPTION,
        state_id: str = "state-backlog",
        confirms_readback: bool = True,
        rollback_succeeds: bool = True,
    ) -> None:
        self._description = description
        self._state_id = state_id
        self._confirms = confirms_readback
        self._rollback_succeeds = rollback_succeeds
        self._written = False
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object] | None:
        if ticket_id != _TICKET:
            return None
        state: dict[str, object] = {"name": "Backlog", "type": "backlog"}
        if self._state_id:
            state["id"] = self._state_id
        return {
            "id": "issue-16025",
            "identifier": _TICKET,
            "state": state,
            "labels": {"nodes": []},
            "team": {"id": "team-1"},
            "description": self._description,
            "children": {"nodes": []},
        }

    async def fetch_done_state_id(self, team_id: str) -> str | None:
        return "state-done"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        if state_id == "state-done":
            self._written = True
            return True
        return self._rollback_succeeds

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        base: list[dict[str, object]] = [
            {
                "id": "554bf26e-349b-4f1a-9385-6a0e5b5b2600",
                "createdAt": "2026-08-14T13:34:06Z",
                "actorId": "human",
                "fromState": {"type": "unstarted"},
                "toState": {"type": "backlog"},
            }
        ]
        if self._written and self._confirms:
            return [
                {
                    "id": "flip-entry",
                    "createdAt": "2026-09-06T21:42:17Z",
                    "actorId": None,
                    "fromState": {"type": "backlog"},
                    "toState": {"type": "completed"},
                },
                *base,
            ], ""
        return base, ""


def _gh(*, workflow_runs: list[dict[str, object]] | None, error: str = ""):
    """A gh seam that also answers the gate-probe workflow-runs read."""
    calls: list[str] = []

    async def run_gh(args: list[str], timeout: float) -> tuple[Any, str]:
        path = args[2]
        calls.append(path)
        if "/actions/workflows/" in path:
            if workflow_runs is None:
                return None, error or "gh api failed"
            return {"workflow_runs": workflow_runs}, ""
        if "/files" in path:
            return [{"filename": f"contracts/{_TICKET}.yaml"}], ""
        if "/pulls/" in path:
            return {"state": "closed", "merged_at": "2026-09-06T20:24:25Z"}, ""
        page = int(path.rsplit("page=", 1)[1])
        return ([_companion(8436, _TICKET)], "") if page == 1 else ([], "")

    run_gh.calls = calls  # type: ignore[attr-defined]
    return run_gh


def _dod(receipt: dict[str, object]):
    async def run_dod(
        ticket_id: str, cwd: str, timeout: int
    ) -> tuple[dict[str, object], int, str]:
        return receipt, 0, ""

    return run_dod


def _handler(
    linear: FakeLinear,
    *,
    receipt: dict[str, object] | None = None,
    workflow_runs: list[dict[str, object]] | None = None,
    gh_error: str = "",
) -> tuple[HandlerEvidenceAutocloseSweep, Any]:
    gh = _gh(workflow_runs=workflow_runs, error=gh_error)
    handler = HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=gh,
        run_dod_verify_command=_dod(receipt or _receipt()),
    )
    return handler, gh


def _request(**overrides: object) -> ModelEvidenceAutocloseSweepRequest:
    payload: dict[str, object] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 24,
        "apply": True,
        "backfill_lookback_hours": 0,
        "readback_delay_seconds": 0,
        "readback_max_attempts": 2,
    }
    payload.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**payload)


# ---------------------------------------------------------------------------
# (a) An UNCONFIRMED readback is a refusal, never an apply.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAnUnconfirmedReadbackIsARefusal:
    async def test_the_write_is_rolled_back_to_the_pre_write_state(self) -> None:
        """The label and the flip cannot coexist on the board.

        Before this, run 34061364537 left OMN-16025 reading Done under a
        comment that said the readback did not confirm. Every downstream sweep
        and rollup reads the board; almost nobody opens the comment.
        """
        linear = FakeLinear(confirms_readback=False)
        handler, _ = _handler(
            linear, receipt=_receipt(total=7, verified=6, non_probative=1)
        )
        # OMN-18056: the first eligible observation arms the re-draw and
        # writes no Done. Everything this class asserts lives on the WRITE
        # path, which is the second tick.
        await handler.handle(_request())
        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert (
            outcome.decision is EnumEvidenceAutocloseDecision.ERROR_READBACK_UNCONFIRMED
        )
        assert outcome.flip_rolled_back is True
        assert result.tickets_flipped == 0
        # Two writes: the Done, then the restore. The board ends where it began.
        assert linear.state_updates == [
            ("issue-16025", "state-done"),
            ("issue-16025", "state-backlog"),
        ]
        assert "rolled back" in outcome.reason.lower()

    async def test_the_audit_comment_does_not_claim_a_flip(self) -> None:
        """`class=flipped` is the prior-revert fence's anchor.

        Stamping it on a write that was taken back would teach the fence that
        this mechanism had closed the ticket, and the fence would then refuse a
        LATER, legitimate flip on that ticket.
        """
        linear = FakeLinear(confirms_readback=False)
        handler, _ = _handler(
            linear, receipt=_receipt(total=7, verified=6, non_probative=1)
        )
        # OMN-18056: the first eligible observation arms the re-draw and
        # writes no Done. Everything this class asserts lives on the WRITE
        # path, which is the second tick.
        await handler.handle(_request())
        await handler.handle(_request())

        # Two comments across the two ticks: the re-draw arming, then this
        # refusal. The refusal is the last one.
        assert len(linear.comments) == 2
        body = linear.comments[-1][1]
        assert "class=flipped" not in body
        assert "class=readback_unconfirmed" in body
        assert "REFUSED" in body
        assert "ROLLED BACK" in body
        # The counters and the companion still have to be legible.
        assert "8436" in body

    async def test_a_failed_rollback_says_so_rather_than_claiming_a_refusal(
        self,
    ) -> None:
        """Fail loud: this is the one shape that leaves a Done nobody proved."""
        linear = FakeLinear(confirms_readback=False, rollback_succeeds=False)
        handler, _ = _handler(
            linear, receipt=_receipt(total=7, verified=6, non_probative=1)
        )
        # OMN-18056: the first eligible observation arms the re-draw and
        # writes no Done. Everything this class asserts lives on the WRITE
        # path, which is the second tick.
        await handler.handle(_request())
        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.flip_rolled_back is False
        assert outcome.applied is True, (
            "a real Linear mutation was made; a receipt that denies it is "
            "worse than one that overstates it (OMN-17658)"
        )
        assert "human read" in outcome.reason.lower()
        assert "human read" in linear.comments[-1][1].lower()

    async def test_no_rollback_target_refuses_before_writing_anything(self) -> None:
        """A write this run cannot undo is a write it may not make.

        Discovering the rollback target is missing AFTER the Done is on the
        board leaves exactly the stuck-open Done the rollback exists to
        prevent, so the conjunct is resolved first.
        """
        linear = FakeLinear(confirms_readback=False, state_id="")
        handler, _ = _handler(
            linear, receipt=_receipt(total=7, verified=6, non_probative=1)
        )
        # OMN-18056: the first eligible observation arms the re-draw and
        # writes no Done. Everything this class asserts lives on the WRITE
        # path, which is the second tick.
        await handler.handle(_request())
        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.decision is EnumEvidenceAutocloseDecision.ERROR_LINEAR_API
        assert linear.state_updates == []
        assert "rollback target" in outcome.reason.lower()

    async def test_a_confirmed_readback_still_flips_and_does_not_roll_back(
        self,
    ) -> None:
        """The common path pays nothing for any of the above."""
        linear = FakeLinear(confirms_readback=True)
        handler, _ = _handler(
            linear, receipt=_receipt(total=7, verified=6, non_probative=1)
        )
        # OMN-18056: the first eligible observation arms the re-draw and
        # writes no Done. Everything this class asserts lives on the WRITE
        # path, which is the second tick.
        await handler.handle(_request())
        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
        assert outcome.flip_rolled_back is False
        assert linear.state_updates == [("issue-16025", "state-done")]
        assert result.tickets_flipped == 1


# ---------------------------------------------------------------------------
# (b) The AC-coverage re-read must actually read the criteria.
# ---------------------------------------------------------------------------


class TestAcceptanceCriteriaAreReadWithoutAHeading:
    def test_the_omn_16025_body_parses_its_five_links(self) -> None:
        """It parsed ZERO before: no heading matched, so no item was collected."""
        items = _acceptance_criteria_items(_OMN_16025_DESCRIPTION)
        assert len(items) == 5
        assert items[0].startswith("Intent submitted through the live gateway")
        assert items[4].startswith("Complete ledger chain")

    def test_a_heading_still_scopes_the_section_when_one_exists(self) -> None:
        """The fallback must not widen a body that DOES declare its section."""
        description = (
            "# Background\n\n- a background bullet\n- another one\n\n"
            "## Acceptance criteria\n\n- the only real criterion\n\n"
            "## Notes\n\n- not a criterion\n"
        )
        assert _acceptance_criteria_items(description) == ["the only real criterion"]

    def test_an_empty_body_is_not_a_gap(self) -> None:
        """Linear returns null for a bodyless ticket; that is not a criterion."""
        assert _ac_coverage_gap("", 6, 6, 6) == ("", ())


class TestNonProbativeCoverageDoesNotProveACriterion:
    def test_a_tie_holds_because_a_tie_refutes_nothing(self) -> None:
        """6 == 6 released the flip under the strict form. It should not.

        The bound exists to REFUTE "every criterion is covered by at least one
        verified probative check". A tie is the absence of a majority either
        way, and a refutation bound must not read that as support.
        """
        reason, uncovered = _ac_coverage_gap(_OMN_16025_DESCRIPTION, 6, 6, 6)
        assert reason, "6 verified against 6 non-probative must hold"
        assert "at least half" in reason
        assert len(uncovered) == 5

    def test_a_single_criterion_with_only_non_probative_company_holds(self) -> None:
        description = "## Acceptance\n\n- the one thing this ticket claims\n"
        reason, uncovered = _ac_coverage_gap(description, 1, 1, 1)
        assert reason
        assert uncovered == ("the one thing this ticket claims",)

    def test_a_probative_majority_still_releases_the_flip(self) -> None:
        """OMN-17976's shape — 4 criteria, 4 verified, 2 non-probative.

        The bound is a refutation, not a demand for a clean sweep; a bounded
        minority of provenance entries is exactly what it is meant to tolerate.
        """
        description = "## Acceptance criteria\n\n- one\n- two\n- three\n- four\n"
        assert _ac_coverage_gap(description, 4, 4, 2) == ("", ())


# ---------------------------------------------------------------------------
# (c) A ticket whose own gate probe is red holds.
# ---------------------------------------------------------------------------


class TestGateProbeDeclarationParsing:
    def test_a_full_declaration_resolves(self) -> None:
        assert _gate_probe_declaration(
            "Some prose.\n\nGate: OmniNode-ai/omnibase_infra chain-canary.yml\n"
        ) == ("OmniNode-ai/omnibase_infra", "chain-canary.yml", "")[:2] + (
            "OmniNode-ai/omnibase_infra chain-canary.yml",
        )

    def test_a_bare_workflow_name_is_unresolvable_not_guessed(self) -> None:
        repo, workflow, raw = _gate_probe_declaration("Gate: chain-canary.yml\n")
        assert (repo, workflow) == ("", "")
        assert raw == "chain-canary.yml"

    def test_a_body_with_no_gate_line_declares_no_probe(self) -> None:
        assert _gate_probe_declaration(_OMN_16025_DESCRIPTION) == ("", "", "")


@pytest.mark.asyncio
class TestARedGateProbeHolds:
    _WITH_GATE = (
        "## Acceptance criteria\n\n- AC1: one\n- AC2: two\n\n"
        "Gate: OmniNode-ai/omnibase_infra chain-canary.yml\n"
    )

    async def test_the_newest_completed_run_being_failure_holds(self) -> None:
        """chain-canary was `failure` on every run of 2026-09-06, including one
        fired two minutes after the flip."""
        linear = FakeLinear(description=self._WITH_GATE)
        handler, _ = _handler(
            linear,
            receipt=_receipt(total=6, verified=4, non_probative=2),
            workflow_runs=[{"id": 34061981317, "conclusion": "failure"}],
        )
        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.decision is EnumEvidenceAutocloseDecision.SKIPPED_GATE_PROBE_RED
        assert outcome.gate_probe_conclusion == "failure"
        assert "34061981317" in outcome.reason
        assert linear.state_updates == [], "a hold writes nothing"
        assert result.tickets_skipped == 1

    async def test_a_green_probe_releases_the_flip(self) -> None:
        linear = FakeLinear(description=self._WITH_GATE)
        handler, _ = _handler(
            linear,
            receipt=_receipt(total=6, verified=4, non_probative=2),
            workflow_runs=[{"id": 1, "conclusion": "success"}],
        )
        # OMN-18056: the first tick arms the re-draw, the second flips.
        await handler.handle(_request())
        result = await handler.handle(_request())
        assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED

    async def test_no_completed_run_at_all_holds(self) -> None:
        """Unattempted is not proven."""
        linear = FakeLinear(description=self._WITH_GATE)
        handler, _ = _handler(
            linear,
            receipt=_receipt(total=6, verified=4, non_probative=2),
            workflow_runs=[],
        )
        result = await handler.handle(_request())
        assert (
            result.outcomes[0].decision
            is EnumEvidenceAutocloseDecision.SKIPPED_GATE_PROBE_RED
        )

    async def test_an_unreadable_probe_fails_closed(self) -> None:
        linear = FakeLinear(description=self._WITH_GATE)
        handler, _ = _handler(
            linear,
            receipt=_receipt(total=6, verified=4, non_probative=2),
            workflow_runs=None,
            gh_error="HTTP 404",
        )
        result = await handler.handle(_request())
        outcome = result.outcomes[0]
        assert outcome.decision is EnumEvidenceAutocloseDecision.SKIPPED_GATE_PROBE_RED
        assert "404" in outcome.reason

    async def test_an_unresolvable_declaration_holds_without_calling_gh(self) -> None:
        linear = FakeLinear(
            description=(
                "## Acceptance criteria\n\n- AC1: one\n- AC2: two\n\n"
                "Gate: chain-canary.yml\n"
            )
        )
        handler, gh = _handler(
            linear,
            receipt=_receipt(total=6, verified=4, non_probative=2),
            workflow_runs=[{"id": 1, "conclusion": "success"}],
        )
        result = await handler.handle(_request())
        assert (
            result.outcomes[0].decision
            is EnumEvidenceAutocloseDecision.SKIPPED_GATE_PROBE_RED
        )
        assert not any("/actions/workflows/" in path for path in gh.calls)

    async def test_a_ticket_declaring_no_probe_is_unaffected(self) -> None:
        """Positive control: the hold must not fire on the 99% of tickets that
        name no gate at all."""
        linear = FakeLinear(
            description="## Acceptance criteria\n\n- AC1: one\n- AC2: two\n"
        )
        handler, gh = _handler(
            linear, receipt=_receipt(total=6, verified=4, non_probative=2)
        )
        # OMN-18056: the first tick arms the re-draw, the second flips.
        await handler.handle(_request())
        result = await handler.handle(_request())
        assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED
        assert not any("/actions/workflows/" in path for path in gh.calls)


# ---------------------------------------------------------------------------
# (d) The regression fixture: the exact OMN-16025 shape must HOLD.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOmn16025RegressionFixture:
    async def test_the_exact_measured_counters_and_body_hold(self) -> None:
        """6 verified, 6 non-probative, 12 total, the live body — HOLD.

        This is run 34061364537's input reproduced exactly. It produced a Done
        on the board. It must now produce a hold that writes nothing.

        OMN-18056 changed WHICH conjunct holds it, and the change is the
        measurement. The body's five links carry no `AC<n>` label, and the
        contract behind this verdict declares no `binds_ac` on any check --
        which is true of all 8709 contracts in the corpus. So the binding gate
        reaches it first and names the criteria nothing proves, instead of the
        counting rule reporting a ratio. The counting rule is still correct
        about this body and is asserted directly below, as the control: what
        moved is the order the two refusals are reported in, not whether the
        board moves.
        """
        linear = FakeLinear(description=_OMN_16025_DESCRIPTION)
        handler, _ = _handler(linear, receipt=_receipt(binds_ac=()))
        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
        assert outcome.dod_verify_verified_count == 6
        assert outcome.dod_verify_non_probative_count == 6
        assert outcome.dod_verify_total_checks == 12
        assert outcome.dod_verify_failed_count == 0
        assert result.tickets_flipped == 0
        assert linear.state_updates == [], (
            "the board must not move: this is the write run 34061364537 made "
            "and a person had to undo twelve minutes later"
        )
        # The five links the old parser could not see are named in the receipt,
        # now as the criteria that bind to nothing.
        assert len(outcome.uncovered_acceptance_criteria) == 5

        # CONTROL: the counting rule that used to hold this body still refutes
        # it on its own terms, so the binding gate has replaced no coverage.
        coverage_reason, uncovered = _ac_coverage_gap(_OMN_16025_DESCRIPTION, 6, 6, 6)
        assert coverage_reason
        assert len(uncovered) == 5

    async def test_the_same_shape_with_a_declared_red_probe_also_holds(self) -> None:
        """Belt and braces: had OMN-16025 carried a `Gate:` line, the probe's
        own colour would have held it too — chain-canary run 34059601163,
        `conclusion=failure` at 3 of 5 links proven."""
        # OMN-18056: this case is already a counterfactual -- OMN-16025 never
        # carried a `Gate:` line. It gains a labelled, bound criterion for the
        # same reason: the binding gate sits ahead of the probe, so without one
        # the probe would never be reached and the assertion below would be
        # about the wrong conjunct.
        linear = FakeLinear(
            description=_OMN_16025_DESCRIPTION
            + "\n\nGate: OmniNode-ai/omnibase_infra chain-canary.yml\n"
            + f"\n{BOUND_AC_DESCRIPTION}"
        )
        handler, _ = _handler(
            linear,
            # Counters that clear the AC-coverage bound, so the ONLY thing left
            # to hold this candidate is the probe.
            receipt=_receipt(total=8, verified=6, non_probative=2),
            workflow_runs=[{"id": 34059601163, "conclusion": "failure"}],
        )
        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.decision is EnumEvidenceAutocloseDecision.SKIPPED_GATE_PROBE_RED
        assert "34059601163" in outcome.reason
        assert linear.state_updates == []
