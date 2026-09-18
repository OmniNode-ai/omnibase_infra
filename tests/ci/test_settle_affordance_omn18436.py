# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18436 -- the declared settle budget survives a full-length convergence watch.

THE RECURRENCE. Receipt ``lab-pass-receipt-compose-dev-63cea2aa2e2d...`` (run
35385618114, merge sha ``63cea2aa``, emitted 2026-09-18T19:59:56Z) carries::

    settle_budget_sufficient: ok=false
    declared 900s for lane compose-dev; this job could afford 898s
    (ceiling 2700s - elapsed 1682s - reserved tail 120s); granted 898s
    -- short by 2s

Two seconds, on a run whose lane answered every readiness endpoint after 101s.
OMN-18436 landed the declared budget precisely so a receipt would stop FAILing
on timing alone, and this receipt FAILed on timing alone anyway.

WHY THE EARLIER FIX DID NOT HOLD. It is not a wrong constant and not a second
budget. ``converge_wall_clock_seconds`` (OMN-18573, PR #3690) bounds the
convergence watch at ``ceiling - elapsed - settle - tail``, i.e. it hands the
watch EVERY second of the ceiling that the settle budget and the tail do not
already claim. Both the converge step and the probe step then measure their own
elapsed time from the same ``LAB_PASS_STARTED_AT`` epoch. So when the watch runs
to its full derived bound -- which is exactly what happens on the INDETERMINATE
path this run took -- the probe step's elapsed read is necessarily LATER than
``elapsed_at_converge + bound``, by the cost of the work between the two reads:
the watch's own return, the step transition, and a ``uv run python`` start to
re-derive the budget. Call that cost d. Then::

    affordable = ceiling - (elapsed + bound + d) - tail
               = ceiling - elapsed - (ceiling - elapsed - settle - tail) - d - tail
               = settle - d

and ``sufficient`` is ``affordable >= declared``, so it is FALSE whenever d > 0,
which is always. The shortfall is not a flake and not a tuning error: it is a
GUARANTEE on every run whose convergence watch runs to its full bound. On this
run d was 2 seconds.

WHAT THESE TESTS PIN.

1. The converge bound reserves the job's own inter-step cost as well as the
   settle budget and the tail, so a full-length watch still leaves the WHOLE
   declared budget affordable. The reserve comes out of ceiling slack that was
   already unallocated (2700 - 1500 - 900 - 120 = 180), so it costs the lane's
   own 25-minute grant nothing -- asserted below rather than asserted in prose.
2. A shortfall that never bit does not fail a receipt. When the lane answered
   every endpoint inside the grant, a short grant proves nothing about the lane
   and the check says so. AC4's ambiguity case is unchanged and is the positive
   control here: a lane that did NOT answer inside a short grant still fails,
   because then "unhealthy" and "out of clock" really are indistinguishable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.ci.lab_pass_receipt import (
    SETTLE_BUDGET_CHECK,
    ModelSettleBudget,
    ModelSettleOutcome,
    settle_budget_check,
)
from scripts.ci.lane_settle_budget import (
    STEP_OVERHEAD_SECONDS,
    affordable_seconds,
    converge_wall_clock_seconds,
    derive_settle_budget,
    load_declaration,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LANE = "compose-dev"

WORKFLOWS = (
    REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml",
    REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger-reusable.yml",
)

#: Every number below is read off the receipt named in this module's docstring,
#: not chosen. They are the measurement this fix is accountable to.
CEILING_SECONDS = 2700
RESERVED_TAIL_SECONDS = 120
MEASURED_ELAPSED_AT_PROBE = 1682
MEASURED_AFFORDABLE = 898
MEASURED_SHORTFALL = 2
MEASURED_BOOT_SECONDS = 101.0


def _verify_job(path: Path) -> dict[str, object]:
    jobs = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]
    for name, job in jobs.items():
        if "verify" in name and "converged" in name:
            return dict(job)
    msg = f"{path.name} declares no verify-converged job"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# The arithmetic: a full-length watch leaves the declared budget intact
# ---------------------------------------------------------------------------
class TestAFullLengthWatchStillLeavesTheDeclaredBudget:
    @pytest.mark.parametrize("inter_step_cost", [0, 1, 2, 15, 60])
    def test_the_probe_can_afford_the_declaration_after_the_watch_runs_out(
        self, inter_step_cost: int
    ) -> None:
        """The property the recurrence violated, over the whole plausible range
        of the cost between the two elapsed reads. 2 is the measured value; 60
        is the reserve, so this is a boundary test as well as a range test."""
        bound = converge_wall_clock_seconds(
            lane=LANE,
            job_ceiling_seconds=CEILING_SECONDS,
            elapsed_seconds=0,
            reserved_tail_seconds=RESERVED_TAIL_SECONDS,
        )
        elapsed_at_probe = bound + inter_step_cost
        budget = derive_settle_budget(
            LANE, CEILING_SECONDS, elapsed_at_probe, RESERVED_TAIL_SECONDS
        )
        assert budget.sufficient, (
            f"a watch that ran its full {bound}s bound and cost "
            f"{inter_step_cost}s to return left only "
            f"{budget.affordable_seconds}s against a declared "
            f"{budget.declared_seconds}s. That is the OMN-18436 recurrence: the "
            "receipt FAILs on timing alone and rule 24(b) refuses the sha."
        )
        assert budget.granted_seconds == budget.declared_seconds

    def test_the_receipt_s_own_numbers_replayed(self) -> None:
        """The recurrence, by its measured values rather than by its shape."""
        # The pre-change arithmetic, kept as the incident record. This is what
        # the emitted receipt carried, and it is still true of those inputs.
        assert (
            affordable_seconds(
                CEILING_SECONDS, MEASURED_ELAPSED_AT_PROBE, RESERVED_TAIL_SECONDS
            )
            == MEASURED_AFFORDABLE
        )
        declared = load_declaration(LANE).settle_budget_seconds
        assert declared - MEASURED_AFFORDABLE == MEASURED_SHORTFALL

        # And the same run under the fix: the watch stops one reserve earlier,
        # so the probe reads its elapsed time with the declaration still whole.
        bound = converge_wall_clock_seconds(
            lane=LANE,
            job_ceiling_seconds=CEILING_SECONDS,
            elapsed_seconds=0,
            reserved_tail_seconds=RESERVED_TAIL_SECONDS,
        )
        elapsed_at_probe = bound + MEASURED_SHORTFALL
        budget = derive_settle_budget(
            LANE, CEILING_SECONDS, elapsed_at_probe, RESERVED_TAIL_SECONDS
        )
        assert budget.sufficient
        assert budget.granted_seconds == declared

    def test_the_reserve_is_real_and_is_not_zero(self) -> None:
        """A reserve of zero reinstates the guarantee this ticket removed, and
        would do it silently: every assertion above still passes at d=0."""
        assert STEP_OVERHEAD_SECONDS > 0
        assert converge_wall_clock_seconds(
            lane=LANE,
            job_ceiling_seconds=CEILING_SECONDS,
            elapsed_seconds=0,
            reserved_tail_seconds=RESERVED_TAIL_SECONDS,
        ) == (
            CEILING_SECONDS
            - load_declaration(LANE).settle_budget_seconds
            - RESERVED_TAIL_SECONDS
            - STEP_OVERHEAD_SECONDS
        )

    def test_a_spent_ceiling_still_yields_zero_rather_than_a_negative_bound(
        self,
    ) -> None:
        assert (
            converge_wall_clock_seconds(
                lane=LANE,
                job_ceiling_seconds=CEILING_SECONDS,
                elapsed_seconds=10_000,
                reserved_tail_seconds=RESERVED_TAIL_SECONDS,
            )
            == 0
        )


class TestTheReserveIsPaidOutOfSlackAndNotOutOfTheLanesGrant:
    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_the_bound_still_covers_the_lane_s_own_wait_timeout(
        self, path: Path
    ) -> None:
        """``--wait-timeout 25m`` is the LANE's grant, measured from deploy-agent
        acceptance. The reserve must come out of ceiling slack, never out of
        that: a bound below the lane's grant would shorten the lane's clock to
        pay for the job's overhead, which is the trade OMN-18573 removed."""
        text = path.read_text(encoding="utf-8")
        import re

        match = re.search(r"--wait-timeout\s+(\d+)m", text)
        assert match is not None, f"{path.name} declares no convergence wait timeout"
        lane_grant = int(match.group(1)) * 60
        ceiling = int(_verify_job(path)["timeout-minutes"]) * 60  # type: ignore[call-overload]
        bound = converge_wall_clock_seconds(
            lane=LANE,
            job_ceiling_seconds=ceiling,
            elapsed_seconds=0,
            reserved_tail_seconds=RESERVED_TAIL_SECONDS,
        )
        assert bound >= lane_grant, (
            f"{path.name}: the derived watch bound {bound}s is below the lane's "
            f"own {lane_grant}s grant, so the reserve is being paid out of the "
            "lane's clock instead of out of ceiling slack."
        )

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_the_ceiling_covers_every_declared_part_including_the_reserve(
        self, path: Path
    ) -> None:
        import re

        text = path.read_text(encoding="utf-8")
        match = re.search(r"--wait-timeout\s+(\d+)m", text)
        assert match is not None
        required = (
            int(match.group(1)) * 60
            + load_declaration(LANE).settle_budget_seconds
            + RESERVED_TAIL_SECONDS
            + STEP_OVERHEAD_SECONDS
        )
        ceiling = int(_verify_job(path)["timeout-minutes"]) * 60  # type: ignore[call-overload]
        assert ceiling >= required, (
            f"{path.name}: ceiling {ceiling}s cannot contain its declared parts "
            f"plus the {STEP_OVERHEAD_SECONDS}s inter-step reserve ({required}s)."
        )


# ---------------------------------------------------------------------------
# A shortfall that never bit is not a finding
# ---------------------------------------------------------------------------
def _short_budget() -> ModelSettleBudget:
    """The budget the recurrence receipt actually carried."""
    return ModelSettleBudget(
        lane=LANE,
        declared_seconds=900,
        affordable_seconds=MEASURED_AFFORDABLE,
        job_ceiling_seconds=CEILING_SECONDS,
        elapsed_seconds=MEASURED_ELAPSED_AT_PROBE,
        reserved_tail_seconds=RESERVED_TAIL_SECONDS,
        source="config/lab_pass_settle_budget.yaml",
    )


class TestAShortGrantThatTheLaneBeatIsNotAFailure:
    def test_a_lane_that_answered_inside_the_grant_passes_the_check(self) -> None:
        """101s of a 898s grant. The 2s the grant was short of the declaration
        could not have changed the answer, and a check that fails anyway fails
        the whole receipt on timing alone -- the shape OMN-18436 exists to
        remove."""
        check = settle_budget_check(
            _short_budget(),
            outcome=ModelSettleOutcome(
                ready=True,
                waited_seconds=MEASURED_BOOT_SECONDS,
                granted_seconds=float(MEASURED_AFFORDABLE),
                pending=(),
            ),
        )
        assert check.name == SETTLE_BUDGET_CHECK
        assert check.ok is True

    def test_it_still_names_both_numbers_and_the_shortfall(self) -> None:
        """Passing is not the same as silent. The receipt still records that
        the job could not afford the declaration, so the ceiling stays
        diagnosable from the artifact alone."""
        check = settle_budget_check(
            _short_budget(),
            outcome=ModelSettleOutcome(
                ready=True,
                waited_seconds=MEASURED_BOOT_SECONDS,
                granted_seconds=float(MEASURED_AFFORDABLE),
                pending=(),
            ),
        )
        assert "900" in check.evidence
        assert str(MEASURED_AFFORDABLE) in check.evidence
        assert f"short by {MEASURED_SHORTFALL}s" in check.evidence
        assert "101s" in check.evidence

    def test_the_ambiguity_case_is_unchanged_and_still_fails(self) -> None:
        """AC4's positive control. A lane that did NOT answer inside a short
        grant is exactly the case where "unhealthy" and "out of clock" cannot
        be told apart, and that is still a failure."""
        check = settle_budget_check(
            _short_budget(),
            outcome=ModelSettleOutcome(
                ready=False,
                waited_seconds=float(MEASURED_AFFORDABLE),
                granted_seconds=float(MEASURED_AFFORDABLE),
                pending=("http://host.docker.internal:8086/ready",),
            ),
        )
        assert check.ok is False
        assert "run out of CLOCK" in check.evidence

    def test_with_no_outcome_recorded_the_verdict_is_the_bare_affordability(
        self,
    ) -> None:
        """A caller that knows nothing about the wait gets AC4's original
        verdict. Absence of an outcome is not evidence that the lane answered."""
        check = settle_budget_check(_short_budget())
        assert check.ok is False

    def test_a_sufficient_grant_passes_whatever_the_lane_did(self) -> None:
        budget = derive_settle_budget(
            LANE, CEILING_SECONDS, 1000, RESERVED_TAIL_SECONDS
        )
        assert budget.sufficient
        for outcome in (
            ModelSettleOutcome(
                ready=False, waited_seconds=900.0, granted_seconds=900.0, pending=("x",)
            ),
            ModelSettleOutcome(
                ready=True, waited_seconds=101.0, granted_seconds=900.0, pending=()
            ),
        ):
            assert settle_budget_check(budget, outcome=outcome).ok is True
