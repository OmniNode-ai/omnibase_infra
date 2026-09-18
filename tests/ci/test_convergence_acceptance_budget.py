# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18573 -- the convergence budget is measured from deploy-agent acceptance.

WHAT THESE PIN
--------------
``verify-lane-converged`` waits for the ``.201`` compose dev lane to report a
revision containing the merge sha, and the ``deployed_revision`` check of the
rule 24(a) lab-pass receipt carries that verdict. Before this change the wait
was a fixed wall clock started when the STEP started, so every second the
redeploy-start effect spent queueing came out of the budget the LANE was meant
to get. A lane that converged three minutes after the clock expired receipted
``FAIL``, and rule 24(b) then refused a good sha for staging delivery.

Measured, twice: OMN-17214 on 2026-09-16 (lane ``dev-lane-flap-1200``), and
again on 2026-09-17 for merge ``50c57653d3cf484bfd9b8a06e77d41cbf4329dd9``,
receipt artifact ``10496150688``, ``deployed_revision`` the only failing check
of eight, lane converged three minutes later.

THE THREE SHAPES, and why the third is not a FAIL
-------------------------------------------------
* a long queue followed by a convergence inside the budget measured FROM
  ACCEPTANCE is a PASS, even though the same convergence is past a wall clock
  started at the step;
* a lane that has had its whole budget since acceptance and still does not
  contain the merge sha is a FAIL, unchanged;
* an acceptance timestamp that cannot be established at all is INDETERMINATE.
  It is not a FAIL because nothing has been shown about the lane -- the lane
  was never granted a budget to miss. It still makes the receipt non-PASS, so
  the delivery gate stays closed; what changes is what the receipt SAYS.

The budget is never widened to absorb a queue. The ceiling is untouched
(OMN-18573 AC6) and a job that runs out of its own clock before the lane's
budget expires reports that, rather than blaming the lane.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.check_dev_lane_staleness import (
    AcceptanceUnresolvedError,
    Ancestry,
    EnumConvergenceOutcome,
    LaneRevision,
    ModelAcceptanceProbe,
    ModelAgentAcceptance,
    ModelConvergenceBudget,
    convergence_check_outcome,
    convergence_evidence,
    read_agent_acceptance,
    run_convergence_wait,
)
from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    EnumLabPassCheckOutcome,
    EnumLabPassResult,
    ModelLabPassCheck,
    ModelLabPassReceipt,
    evaluate_gate,
    parse_check_argument,
)

pytestmark = pytest.mark.unit

MERGE_SHA = "50c57653d3cf484bfd9b8a06e77d41cbf4329dd9"
STALE_SHA = "843fe808cc7891a1b2c3d4e5f60718293a4b5c6d"
CORRELATION_ID = "0a3d0f1e-1111-4c2a-9f3b-2a6c8d4e5f60"
T0 = datetime(2026, 9, 17, 11, 0, 0, tzinfo=UTC)
DECLARED = timedelta(minutes=25)

# The wall clock the verify job can actually afford: its 45-minute ceiling less
# the declared settle budget and the reserved tail. Not a number this change
# may move -- see AC6.
WALL_CLOCK = timedelta(seconds=45 * 60 - 900 - 120)


def _lane(revision: str) -> LaneRevision:
    return LaneRevision(
        revision=revision,
        compose_project="omnibase-infra",
        build_source="workspace",
        state="running",
    )


def _contained_ancestry() -> Ancestry:
    return Ancestry(
        relation="identical", commits_ahead=0, observed_on_branch=True, branch="dev"
    )


def _stale_ancestry() -> Ancestry:
    return Ancestry(
        relation="ancestor", commits_ahead=0, observed_on_branch=True, branch="dev"
    )


class _Clock:
    """A monotonic wall clock the wait loop reads instead of ``time``."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _wait(
    *,
    clock: _Clock,
    converges_at: datetime | None,
    acceptance_at: datetime | None,
    acceptance_reason: str = "",
    acceptance_visible_from: datetime | None = None,
    wall_clock: timedelta = WALL_CLOCK,
):
    """Drive the real loop over an injected clock, lane and agent."""

    def read_lane() -> LaneRevision:
        if converges_at is not None and clock.now >= converges_at:
            return _lane(MERGE_SHA)
        return _lane(STALE_SHA)

    def resolve_ancestry(observed: str) -> Ancestry | None:
        return _contained_ancestry() if observed == MERGE_SHA else _stale_ancestry()

    def resolve_acceptance() -> ModelAcceptanceProbe:
        if acceptance_at is None:
            return ModelAcceptanceProbe(acceptance=None, reason=acceptance_reason)
        visible = acceptance_visible_from or acceptance_at
        if clock.now < visible:
            return ModelAcceptanceProbe(
                acceptance=None,
                reason=(
                    f"the deploy agent reports no job for correlation "
                    f"{CORRELATION_ID} yet"
                ),
            )
        return ModelAcceptanceProbe(
            acceptance=ModelAgentAcceptance(
                correlation_id=CORRELATION_ID,
                accepted_at=acceptance_at,
                source=f"http://host.docker.internal:8098/job/{CORRELATION_ID}",
            ),
            reason="",
        )

    return run_convergence_wait(
        expected_revision=MERGE_SHA,
        read_lane=read_lane,
        resolve_ancestry=resolve_ancestry,
        resolve_acceptance=resolve_acceptance,
        declared_budget=DECLARED,
        wall_clock=wall_clock,
        poll_interval=timedelta(seconds=60),
        clock=clock,
        sleep=clock.sleep,
    )


# ---------------------------------------------------------------------------
# AC1 -- a long queue followed by a fast convergence is a PASS
# ---------------------------------------------------------------------------
def test_long_queue_then_convergence_inside_the_acceptance_budget_passes() -> None:
    """The defect, inverted.

    Acceptance lands 1400s into the step, and the lane converges 150s later.
    A wall clock started at the step expires at 1500s and calls that
    NOT_CONVERGED; a budget measured from acceptance still has 1350s left.
    The job can afford the wait (1550s < 1680s), so nothing is widened.
    """
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=T0 + timedelta(seconds=1400),
        converges_at=T0 + timedelta(seconds=1550),
    )

    assert result.outcome is EnumConvergenceOutcome.OK
    assert result.reason == ""
    # The convergence is genuinely past the old fixed bound; without that the
    # test would pass on the pre-change tree too.
    assert result.waited > DECLARED
    assert result.budget.acceptance is not None
    assert result.budget.acceptance.accepted_at == T0 + timedelta(seconds=1400)


def test_passing_evidence_names_the_acceptance_timestamp_and_elapsed() -> None:
    """AC7's shape, asserted on the evidence the receipt will carry."""
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=T0 + timedelta(seconds=1400),
        converges_at=T0 + timedelta(seconds=1550),
    )

    evidence = convergence_evidence(
        lane=result.lane,
        expected_revision=MERGE_SHA,
        ancestry=result.ancestry,
        waited=result.waited,
        converged=True,
        budget=result.budget,
        now=clock.now,
    )

    assert "2026-09-17T11:23:20" in evidence, evidence
    assert "since the deploy agent accepted" in evidence, evidence
    assert "\n" not in evidence


# ---------------------------------------------------------------------------
# AC2 -- a lane that had its whole budget and did not converge is still a FAIL
# ---------------------------------------------------------------------------
def test_budget_exhausted_since_acceptance_without_convergence_is_a_fail() -> None:
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=T0 + timedelta(seconds=10),
        converges_at=None,
    )

    assert result.outcome is EnumConvergenceOutcome.FAIL
    # It stopped on the LANE's budget, not on the job's clock.
    assert result.waited < WALL_CLOCK
    assert result.waited >= DECLARED


def test_failing_evidence_names_the_acceptance_timestamp_and_elapsed() -> None:
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=T0 + timedelta(seconds=10),
        converges_at=None,
    )

    evidence = convergence_evidence(
        lane=result.lane,
        expected_revision=MERGE_SHA,
        ancestry=result.ancestry,
        waited=result.waited,
        converged=False,
        budget=result.budget,
        now=clock.now,
    )

    assert "2026-09-17T11:00:10" in evidence, evidence
    assert "since the deploy agent accepted" in evidence, evidence
    assert STALE_SHA[:12] in evidence
    assert MERGE_SHA[:12] in evidence


def test_an_acceptance_older_than_the_budget_fails_on_the_first_look() -> None:
    """The command was accepted while the RUN was still queued.

    The lane has already had more than its whole budget, so there is nothing
    left to wait for and the verdict is available immediately.
    """
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=T0 - timedelta(seconds=2000),
        converges_at=None,
    )

    assert result.outcome is EnumConvergenceOutcome.FAIL
    assert result.waited < timedelta(seconds=120)


# ---------------------------------------------------------------------------
# AC3 -- an unestablished budget is INDETERMINATE, never a FAIL
# ---------------------------------------------------------------------------
def test_no_correlation_id_is_indeterminate_not_a_lane_failure() -> None:
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=None,
        acceptance_reason=(
            "the publishing job recorded no correlation id for this run, so the "
            "deploy agent's acceptance cannot be located"
        ),
        converges_at=None,
    )

    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert "no correlation id" in result.reason


def test_unreachable_agent_is_indeterminate_and_names_the_transport() -> None:
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=None,
        acceptance_reason="URLError: [Errno 111] Connection refused",
        converges_at=None,
    )

    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert "Connection refused" in result.reason


def test_agent_never_accepts_within_the_window_is_indeterminate() -> None:
    """A 404 for the whole window means the effect never handed it over.

    That is a statement about the hop BEFORE the agent, not about the lane, so
    it is not a lane failure.
    """
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=T0 + timedelta(days=1),
        acceptance_visible_from=T0 + timedelta(days=1),
        converges_at=None,
    )

    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert "no job for correlation" in result.reason
    assert result.budget.acceptance is None


def test_job_clock_exhausted_before_the_lane_budget_is_indeterminate() -> None:
    """Acceptance is known but the job cannot afford the rest of the budget.

    The honest answer is that this run did not observe the lane long enough,
    not that the lane failed. Both numbers are named, the way the settle
    budget's own shortfall check names both.
    """
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=T0 + timedelta(seconds=1000),
        converges_at=None,
        wall_clock=timedelta(seconds=1200),
    )

    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert "could NOT afford" in result.reason
    assert "1500" in result.reason


def test_indeterminate_evidence_names_the_sha_and_the_reason() -> None:
    clock = _Clock(T0)
    result = _wait(
        clock=clock,
        acceptance_at=None,
        acceptance_reason="URLError: [Errno 111] Connection refused",
        converges_at=None,
    )

    evidence = convergence_evidence(
        lane=result.lane,
        expected_revision=MERGE_SHA,
        ancestry=result.ancestry,
        waited=result.waited,
        converged=False,
        budget=result.budget,
        now=clock.now,
        indeterminate_reason=result.reason,
    )

    assert MERGE_SHA[:12] in evidence
    assert "INDETERMINATE" in evidence
    assert "Connection refused" in evidence
    assert "\n" not in evidence


def test_convergence_check_outcome_maps_the_three_verdicts() -> None:
    assert (
        convergence_check_outcome(EnumConvergenceOutcome.OK)
        is EnumLabPassCheckOutcome.PASS
    )
    assert (
        convergence_check_outcome(EnumConvergenceOutcome.FAIL)
        is EnumLabPassCheckOutcome.FAIL
    )
    assert (
        convergence_check_outcome(EnumConvergenceOutcome.INDETERMINATE)
        is EnumLabPassCheckOutcome.INDETERMINATE
    )


# ---------------------------------------------------------------------------
# read_agent_acceptance -- the reader, and its refusals
# ---------------------------------------------------------------------------
def test_read_agent_acceptance_parses_the_agents_own_record() -> None:
    body = json.dumps(
        {
            "correlation_id": CORRELATION_ID,
            "status": "in_progress",
            "accepted_at": "2026-09-17T11:23:20+00:00",
            "completed_at": None,
        }
    )
    acceptance = read_agent_acceptance(
        agent_url="http://host.docker.internal:8098",
        correlation_id=CORRELATION_ID,
        opener=lambda url, timeout: (200, body),
    )
    assert acceptance.accepted_at == datetime(2026, 9, 17, 11, 23, 20, tzinfo=UTC)
    assert acceptance.correlation_id == CORRELATION_ID
    assert CORRELATION_ID in acceptance.source


@pytest.mark.parametrize(
    ("status", "body", "fragment"),
    [
        (404, '{"error": "not found"}', "no job for correlation"),
        (500, '{"error": "boom"}', "HTTP 500"),
        (200, "not json at all", "unreadable"),
        (200, '{"correlation_id": "x"}', "accepted_at"),
        (200, '{"accepted_at": "not-a-timestamp"}', "accepted_at"),
    ],
)
def test_read_agent_acceptance_refuses_rather_than_guessing(
    status: int, body: str, fragment: str
) -> None:
    with pytest.raises(AcceptanceUnresolvedError) as excinfo:
        read_agent_acceptance(
            agent_url="http://host.docker.internal:8098",
            correlation_id=CORRELATION_ID,
            opener=lambda url, timeout: (status, body),
        )
    assert fragment in str(excinfo.value)


def test_read_agent_acceptance_reports_a_transport_failure_as_unresolved() -> None:
    def _boom(url: str, timeout: float) -> tuple[int, str]:
        raise urllib.error.URLError("[Errno 111] Connection refused")

    with pytest.raises(AcceptanceUnresolvedError) as excinfo:
        read_agent_acceptance(
            agent_url="http://host.docker.internal:8098",
            correlation_id=CORRELATION_ID,
            opener=_boom,
        )
    assert "Connection refused" in str(excinfo.value)


def test_read_agent_acceptance_refuses_a_correlation_id_that_is_not_a_uuid() -> None:
    with pytest.raises(AcceptanceUnresolvedError) as excinfo:
        read_agent_acceptance(
            agent_url="http://host.docker.internal:8098",
            correlation_id="../../etc/passwd",
            opener=lambda url, timeout: (200, "{}"),
        )
    assert "correlation id" in str(excinfo.value)


def test_a_budget_cannot_claim_both_an_acceptance_and_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        ModelConvergenceBudget(
            declared_seconds=1500,
            wall_clock_seconds=1680,
            acceptance=ModelAgentAcceptance(
                correlation_id=CORRELATION_ID, accepted_at=T0, source="x"
            ),
            unresolved_reason="but also unresolved",
        )


def test_a_budget_with_no_acceptance_must_carry_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        ModelConvergenceBudget(
            declared_seconds=1500,
            wall_clock_seconds=1680,
            acceptance=None,
            unresolved_reason="",
        )


# ---------------------------------------------------------------------------
# AC4 -- an INDETERMINATE check keeps the receipt non-PASS
# ---------------------------------------------------------------------------
def _receipt(checks: tuple[ModelLabPassCheck, ...], result: EnumLabPassResult):
    return ModelLabPassReceipt(
        sha=MERGE_SHA,
        lane=EnumLabLane.COMPOSE_DEV,
        started_at=T0,
        finished_at=T0 + timedelta(minutes=30),
        result=result,
        checks=checks,
        agent_command_id=CORRELATION_ID,
    )


def test_a_receipt_carrying_an_indeterminate_check_is_not_a_pass() -> None:
    checks = (
        ModelLabPassCheck(name="ready_main", ok=True, evidence="HTTP 200"),
        ModelLabPassCheck.indeterminate_check(
            name="deployed_revision",
            evidence="INDETERMINATE: the deploy agent's acceptance could not be read",
        ),
    )
    receipt = _receipt(checks, EnumLabPassResult.FAIL)
    assert receipt.result is EnumLabPassResult.FAIL
    assert not all(c.ok for c in receipt.checks)


def test_a_pass_receipt_over_an_indeterminate_check_is_refused() -> None:
    checks = (
        ModelLabPassCheck(name="ready_main", ok=True, evidence="HTTP 200"),
        ModelLabPassCheck.indeterminate_check(
            name="deployed_revision", evidence="INDETERMINATE: agent unreachable"
        ),
    )
    with pytest.raises(ValueError, match="deployed_revision"):
        _receipt(checks, EnumLabPassResult.PASS)


def test_an_indeterminate_check_cannot_also_be_ok() -> None:
    with pytest.raises(ValueError, match="indeterminate"):
        ModelLabPassCheck(
            name="deployed_revision", ok=True, evidence="e", indeterminate=True
        )


def test_an_indeterminate_check_round_trips_through_the_wire() -> None:
    check = ModelLabPassCheck.indeterminate_check(
        name="deployed_revision", evidence="INDETERMINATE: agent unreachable"
    )
    payload = check.to_dict()
    assert payload["ok"] is False
    assert payload["outcome"] == EnumLabPassCheckOutcome.INDETERMINATE.value
    assert ModelLabPassCheck.from_dict(payload) == check


def test_an_ordinary_check_is_byte_identical_to_the_pre_change_wire_form() -> None:
    """The extra field is written only when it says something ``ok`` cannot.

    A reader that predates this change then refuses exactly the receipts it
    could not have interpreted, and parses unchanged every receipt it could.
    """
    for ok in (True, False):
        payload = ModelLabPassCheck(name="ready_main", ok=ok, evidence="e").to_dict()
        assert set(payload) == {"name", "ok", "evidence"}


def test_the_check_argument_parser_accepts_the_indeterminate_verdict() -> None:
    check = parse_check_argument("deployed_revision:indeterminate:agent unreachable")
    assert check.outcome is EnumLabPassCheckOutcome.INDETERMINATE
    assert check.ok is False
    assert check.evidence == "agent unreachable"


# ---------------------------------------------------------------------------
# AC5 -- the delivery gate's refusal reads differently for INDETERMINATE
# ---------------------------------------------------------------------------
class _Out:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str) -> None:
        self.lines.append(text)

    @property
    def text(self) -> str:
        return "".join(self.lines)


def _gate_text(monkeypatch: pytest.MonkeyPatch, receipt: ModelLabPassReceipt) -> str:
    import scripts.ci.lab_pass_receipt as mod

    monkeypatch.setattr(
        mod, "list_artifacts", lambda repo, name: [{"id": 1, "created_at": "2026"}]
    )
    monkeypatch.setattr(mod, "download_receipt", lambda repo, artifact_id: receipt)
    out = _Out()
    rc = evaluate_gate(
        repo="OmniNode-ai/omnibase_infra",
        sha=MERGE_SHA,
        lanes=[receipt.lane],
        out=out,
    )
    assert rc == 1
    return out.text


def test_the_gate_refusal_names_the_sha_and_the_indeterminate_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "INDETERMINATE: the deploy agent's acceptance could not be read"
    receipt = _receipt(
        (
            ModelLabPassCheck(name="ready_main", ok=True, evidence="HTTP 200"),
            ModelLabPassCheck.indeterminate_check(
                name="deployed_revision", evidence=reason
            ),
        ),
        EnumLabPassResult.FAIL,
    )
    text = _gate_text(monkeypatch, receipt)

    assert MERGE_SHA in text
    assert "deployed_revision" in text
    assert reason in text
    assert "INDETERMINATE" in text


def test_the_indeterminate_refusal_is_not_the_same_text_as_a_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing = _receipt(
        (
            ModelLabPassCheck(name="ready_main", ok=True, evidence="HTTP 200"),
            ModelLabPassCheck(
                name="deployed_revision",
                ok=False,
                evidence="lane at abc; not converged",
            ),
        ),
        EnumLabPassResult.FAIL,
    )
    indeterminate = _receipt(
        (
            ModelLabPassCheck(name="ready_main", ok=True, evidence="HTTP 200"),
            ModelLabPassCheck.indeterminate_check(
                name="deployed_revision", evidence="INDETERMINATE: agent unreachable"
            ),
        ),
        EnumLabPassResult.FAIL,
    )

    fail_text = _gate_text(monkeypatch, failing)
    indeterminate_text = _gate_text(monkeypatch, indeterminate)

    assert fail_text != indeterminate_text
    # The generic refusal enumerates every refusable state, so the token alone
    # is not the discriminator -- the INDETERMINATE-specific line is.
    marker = "asserts nothing about the lab lane"
    assert marker not in fail_text
    assert marker in indeterminate_text


# ---------------------------------------------------------------------------
# The wiring, pinned. A fix that lives only in a Python function and is never
# reached by the job is the shape rule 5 calls detection rather than
# enforcement, so the workflow's own flags are asserted here.
# ---------------------------------------------------------------------------
WORKFLOW = REPO_ROOT / ".github/workflows/runtime-rebuild-trigger.yml"


def _converge_step() -> dict:
    import yaml

    model = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = model["jobs"]["verify-lane-converged"]["steps"]
    step = next(s for s in steps if s.get("id") == "converge")
    assert isinstance(step, dict)
    return step


def test_the_convergence_step_is_handed_the_agent_surface_and_the_correlation_id() -> (
    None
):
    body = _converge_step()["run"]
    assert "--agent-url" in body
    assert "--correlation-id" in body
    assert "--wall-clock-seconds" in body
    # The lane's budget is unchanged (AC6): the flags above bound the WAIT, and
    # a fix that quietly widened the grant would show up right here.
    assert "--wait-timeout 25m" in body


def test_the_convergence_wall_clock_is_derived_not_typed() -> None:
    body = _converge_step()["run"]
    assert "lane_settle_budget.py" in body
    assert "--converge-wall-clock" in body


def test_the_trigger_job_publishes_the_correlation_id() -> None:
    import yaml

    model = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    outputs = model["jobs"]["trigger-rebuild"]["outputs"]
    assert "correlation_id" in outputs


def test_the_emit_step_reads_the_three_valued_verdict() -> None:
    import yaml

    model = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = model["jobs"]["verify-lane-converged"]["steps"]
    emit = next(s for s in steps if "emit" in str(s.get("name", "")).lower())
    assert "steps.converge.outputs.verdict" in emit["env"]["CONVERGE_VERDICT"]
    assert "indeterminate" in emit["run"]


def test_the_job_ceiling_is_unchanged_by_this_ticket() -> None:
    """AC6, as a value rather than a promise."""
    import yaml

    model = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert model["jobs"]["verify-lane-converged"]["timeout-minutes"] == 45


def test_the_declared_settle_budget_is_unchanged_by_this_ticket() -> None:
    import yaml

    declaration = yaml.safe_load(
        (REPO_ROOT / "config/lab_pass_settle_budget.yaml").read_text(encoding="utf-8")
    )
    assert declaration["lanes"]["compose-dev"]["settle_budget_seconds"] == 900


def test_the_converge_wall_clock_leaves_room_for_the_settle_and_the_tail() -> None:
    """The bound is derived, and the derivation is the thing under test."""
    from scripts.ci.lane_settle_budget import (
        STEP_OVERHEAD_SECONDS,
        converge_wall_clock_seconds,
    )

    seconds = converge_wall_clock_seconds(
        lane="compose-dev",
        job_ceiling_seconds=45 * 60,
        elapsed_seconds=0,
        reserved_tail_seconds=120,
    )
    # The job's own inter-step cost is reserved too (OMN-18436): without it a
    # watch that runs to this bound leaves the probe less than the declared
    # settle budget, and the affordability check then fails by that cost on a
    # lane it has just proven healthy. Receipt 63cea2aa, short by 2s.
    assert seconds == 45 * 60 - 900 - 120 - STEP_OVERHEAD_SECONDS
    # A job that has already spent its ceiling gets zero, never a negative
    # bound that would read as "wait forever".
    assert (
        converge_wall_clock_seconds(
            lane="compose-dev",
            job_ceiling_seconds=45 * 60,
            elapsed_seconds=10_000,
            reserved_tail_seconds=120,
        )
        == 0
    )


def test_an_unresolvable_declaration_refuses_a_converge_bound(tmp_path: Path) -> None:
    from scripts.ci.lane_settle_budget import (
        SettleBudgetError,
        converge_wall_clock_seconds,
    )

    with pytest.raises(SettleBudgetError):
        converge_wall_clock_seconds(
            lane="compose-dev",
            job_ceiling_seconds=2700,
            elapsed_seconds=0,
            reserved_tail_seconds=120,
            path=tmp_path / "absent.yaml",
        )


# ---------------------------------------------------------------------------
# OMN-18685 -- the same budget, on the SIBLING guard.
#
# OMN-18573 above moved the convergence budget off the step's clock and onto the
# deploy agent's acceptance, in `check_dev_lane_staleness.py`. That guard serves
# the DIRECT caller: an omnibase_infra merge, proven by the container's
# `org.opencontainers.image.revision` label. A SIBLING merge (omnimarket,
# omnibase_core, omnibase_compat) does not move that label at all, so the
# sibling path runs a second guard -- `check_lane_sibling_revision.py`, reading
# `/app/build-provenance.json` -- and the port was never made to it. It kept a
# bare wall clock opened at `main()`.
#
# MEASURED, omnimarket#2641 on 2026-09-18. Receipt artifact 10546422861,
# `lab-pass-receipt-compose-dev-8d52a7ccf563...`: FAIL on `sibling_revision`
# alone, its other seven checks green including every readiness probe. Publish
# 11:57:25Z, guard opened its 25-minute clock 11:58:01Z, the agent did not
# accept until 12:08:41Z (queued behind a job that ran 42m07s), and
# accept-to-recreate on that lane is about 22m15s. The agent's own build argv
# read `OMNIMARKET_REF=8d52a7ccf563...` verbatim -- it resolved exactly the
# right sha. The lane needed ~32m40s from the guard's start and had 25m00s.
# ---------------------------------------------------------------------------

from scripts.ci.check_lane_sibling_revision import (
    EXIT_INDETERMINATE as SIBLING_EXIT_INDETERMINATE,
)
from scripts.ci.check_lane_sibling_revision import (
    ModelSiblingObservation,
    run_sibling_convergence_wait,
    sibling_convergence_evidence,
)

SIBLING_REPO = "omnimarket"
SIBLING_MERGE_SHA = "8d52a7ccf56333ceaf82039cdbe82b6f30653a98"
SIBLING_STALE_SHA = "948be2e1a517d04d63f6d0a1f2e3c4b5a6978899"

#: 11:57:25Z, the moment the trigger published redeploy-start for #2641.
S_PUBLISH = datetime(2026, 9, 18, 11, 57, 25, tzinfo=UTC)
#: 11:58:01Z, the moment the guard opened its clock -- 36s after the publish.
S_GUARD_START = datetime(2026, 9, 18, 11, 58, 1, tzinfo=UTC)
#: 12:08:41Z, the moment the deploy agent accepted the command: 11m16s of queue
#: behind an in-flight 42m07s job, none of it the lane's doing.
S_ACCEPT = datetime(2026, 9, 18, 12, 8, 41, tzinfo=UTC)
#: accept + 22m15s, the measured accept-to-recreate on this lane.
S_RECREATE = S_ACCEPT + timedelta(minutes=22, seconds=15)


def _sibling_wait(
    *,
    clock: _Clock,
    converges_at: datetime | None,
    acceptance_at: datetime | None,
    acceptance_reason: str = "",
    acceptance_visible_from: datetime | None = None,
    wall_clock: timedelta = WALL_CLOCK,
    unreadable_until: datetime | None = None,
):
    """Drive the real sibling loop over an injected clock, lane and agent."""

    def observe() -> ModelSiblingObservation:
        if unreadable_until is not None and clock.now < unreadable_until:
            return ModelSiblingObservation(
                lane_revision="",
                containment="",
                unreadable_reason=(
                    "docker exec omninode-runtime-effects cat "
                    "/app/build-provenance.json failed (exit 1): No such container"
                ),
            )
        if converges_at is not None and clock.now >= converges_at:
            return ModelSiblingObservation(
                lane_revision=SIBLING_MERGE_SHA,
                containment="ahead",
                unreadable_reason="",
            )
        return ModelSiblingObservation(
            lane_revision=SIBLING_STALE_SHA, containment="behind", unreadable_reason=""
        )

    def resolve_acceptance() -> ModelAcceptanceProbe:
        if acceptance_at is None:
            return ModelAcceptanceProbe(acceptance=None, reason=acceptance_reason)
        visible = acceptance_visible_from or acceptance_at
        if clock.now < visible:
            return ModelAcceptanceProbe(
                acceptance=None,
                reason=(
                    f"the deploy agent reports no job for correlation "
                    f"{CORRELATION_ID} yet"
                ),
            )
        return ModelAcceptanceProbe(
            acceptance=ModelAgentAcceptance(
                correlation_id=CORRELATION_ID,
                accepted_at=acceptance_at,
                source=f"http://host.docker.internal:8098/job/{CORRELATION_ID}",
            ),
            reason="",
        )

    return run_sibling_convergence_wait(
        observe=observe,
        resolve_acceptance=resolve_acceptance,
        declared_budget=DECLARED,
        wall_clock=wall_clock,
        poll_interval=timedelta(seconds=60),
        clock=clock,
        sleep=clock.sleep,
    )


def _sibling_evidence(result) -> str:
    return sibling_convergence_evidence(
        repo=SIBLING_REPO, expected_revision=SIBLING_MERGE_SHA, result=result
    )


# --- AC1: the budget starts at acceptance ---------------------------------


def test_a_queue_then_a_convergence_inside_the_lane_budget_passes() -> None:
    """AC1. The shape a step-anchored clock reports as NOT_CONVERGED.

    Acceptance is 11m16s after the guard starts, and the lane comes to vendor
    the merge 27 minutes in -- past a 25-minute bound measured from the step,
    comfortably inside the same 25 minutes measured from acceptance.
    """
    clock = _Clock(S_GUARD_START)
    result = _sibling_wait(
        clock=clock,
        converges_at=S_GUARD_START + timedelta(minutes=27),
        acceptance_at=S_ACCEPT,
    )
    assert result.outcome is EnumConvergenceOutcome.OK
    assert result.waited >= timedelta(minutes=25), (
        "the point of the ticket: the wait ran PAST the old 25-minute bound"
    )
    evidence = _sibling_evidence(result)
    assert SIBLING_MERGE_SHA[:12] in evidence
    assert SIBLING_STALE_SHA[:12] not in evidence
    # A passing receipt that does not say where the lane's clock started cannot
    # be used to check that it started in the right place.
    assert S_ACCEPT.isoformat() in evidence
    assert CORRELATION_ID in evidence


def test_the_same_convergence_is_not_converged_on_a_step_anchored_budget() -> None:
    """The positive control for the test above: the anchor is what changed.

    Same lane, same convergence. The only difference is that the budget is
    anchored at the guard's own start instead of at the agent's acceptance --
    which is the pre-OMN-18685 shape -- and it reports the lane as failing.
    """
    clock = _Clock(S_GUARD_START)
    result = _sibling_wait(
        clock=clock,
        converges_at=S_GUARD_START + timedelta(minutes=27),
        acceptance_at=S_GUARD_START,
    )
    assert result.outcome is EnumConvergenceOutcome.FAIL
    assert result.observation.lane_revision == SIBLING_STALE_SHA


def test_a_lane_granted_its_whole_budget_from_acceptance_still_fails() -> None:
    """The FAIL is unchanged and is still a statement ABOUT THE LANE.

    The wall clock here is wide enough for the lane's whole 25-minute budget to
    actually elapse. That is the only condition under which this guard is
    entitled to say the lane failed, and the next test pins the consequence.
    """
    clock = _Clock(S_GUARD_START)
    result = _sibling_wait(
        clock=clock,
        converges_at=None,
        acceptance_at=S_ACCEPT,
        wall_clock=timedelta(minutes=40),
    )
    assert result.outcome is EnumConvergenceOutcome.FAIL
    assert result.budget.exhausted(result.finished_at)
    evidence = _sibling_evidence(result)
    assert "behind relative to" in evidence or "is behind relative" in evidence
    assert "INDETERMINATE" not in evidence


def test_a_queue_longer_than_the_jobs_spare_clock_can_never_be_a_lane_fail() -> None:
    """A property worth stating plainly, because it is a consequence not a bug.

    The job can watch for 1680s (its 45-minute ceiling less the declared 900s
    settle budget and the 120s tail). Once the agent's acceptance is 11 minutes
    after the guard starts, the lane's 1500s budget outlives the job's own
    clock, so this guard can no longer reach a FAIL at all on that shape -- it
    reports INDETERMINATE and says the job ran out of clock.

    That is the honest answer and it is fail-closed: the receipt is still
    non-PASS and rule 24(b) still refuses the sha. Making the verdict reachable
    again means giving the job a clock that covers the budget, which is the
    ceiling work tracked as OMN-18637, NOT shrinking the lane's budget back
    onto the step.
    """
    clock = _Clock(S_GUARD_START)
    result = _sibling_wait(clock=clock, converges_at=None, acceptance_at=S_ACCEPT)
    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert not result.budget.exhausted(result.finished_at)
    assert "could NOT afford the lane's declared budget" in _sibling_evidence(result)


# --- AC2: no acceptance is INDETERMINATE, never a PASS and never a FAIL ----


@pytest.mark.parametrize(
    ("reason", "fragment"),
    [
        (
            "the publishing job recorded no correlation id for this run, so the "
            "deploy agent's acceptance cannot be located and the lane's "
            "convergence budget has no start",
            "recorded no correlation id",
        ),
        (
            "http://host.docker.internal:8098/job/"
            + CORRELATION_ID
            + " could not be read: URLError: <urlopen error [Errno 111] "
            "Connection refused>",
            "could not be read",
        ),
        (
            "the deploy agent reports no job for correlation "
            + CORRELATION_ID
            + " (http://host.docker.internal:8098/job/"
            + CORRELATION_ID
            + " -> HTTP 404). The command has not been handed to the agent yet, "
            "so the lane's budget has not started.",
            "reports no job for correlation",
        ),
    ],
    ids=["no-correlation-id", "agent-unreachable", "agent-404-all-window"],
)
def test_an_unestablished_acceptance_is_indeterminate(
    reason: str, fragment: str
) -> None:
    """AC2, one case per cause. None of the three is a PASS or a FAIL."""
    clock = _Clock(S_GUARD_START)
    result = _sibling_wait(
        clock=clock, converges_at=None, acceptance_at=None, acceptance_reason=reason
    )
    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert result.reason == reason
    evidence = _sibling_evidence(result)
    assert evidence.startswith("INDETERMINATE:")
    assert fragment in evidence
    assert "asserts nothing about the lane" in evidence
    # It is still not a pass: the receipt's check outcome is non-ok either way.
    check = parse_check_argument(
        f"sibling_revision:{convergence_check_outcome(result.outcome).value}:{evidence}"
    )
    assert check.ok is False
    assert check.outcome is EnumLabPassCheckOutcome.INDETERMINATE


def test_an_acceptance_that_lands_mid_wait_starts_the_budget_retroactively() -> None:
    """The agent usually accepts DURING the wait, and the budget starts then."""
    clock = _Clock(S_GUARD_START)
    result = _sibling_wait(
        clock=clock,
        converges_at=S_GUARD_START + timedelta(minutes=27),
        acceptance_at=S_ACCEPT,
        acceptance_visible_from=S_ACCEPT,
    )
    assert result.outcome is EnumConvergenceOutcome.OK
    assert result.budget.acceptance is not None
    assert result.budget.acceptance.accepted_at == S_ACCEPT


# --- AC3: the 8d52a7ccf timeline, replayed ---------------------------------


def test_the_8d52a7ccf_timeline_no_longer_blames_the_lane() -> None:
    """AC3. The real run, replayed against the real bounds.

    Publish 11:57:25Z, guard start 11:58:01Z, acceptance 12:08:41Z, the lane
    recreated at acceptance + 22m15s = 12:30:56Z. The job can afford 28 minutes
    of watching (its 45-minute ceiling less the declared 900s settle budget and
    the 120s tail), so it still stops before the lane is there.

    What changes is the VERDICT it stops on. The run that produced receipt
    10546422861 wrote FAIL -- `the dev lane still vendors omnimarket at
    948be2e1a517, which is behind` -- a statement that the lane misbehaved,
    about a lane whose budget had 8 minutes left. It is now INDETERMINATE, and
    the evidence says the JOB ran out of clock rather than that the lane failed.
    Rule 24(b) still refuses the sha; what changes is what the refusal says.
    """
    clock = _Clock(S_GUARD_START)
    result = _sibling_wait(clock=clock, converges_at=S_RECREATE, acceptance_at=S_ACCEPT)
    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert not result.budget.exhausted(result.finished_at), (
        "the lane still had budget left when this job had to stop"
    )
    evidence = _sibling_evidence(result)
    assert evidence.startswith("INDETERMINATE:")
    assert "could NOT afford the lane's declared budget" in evidence
    assert "1500s granted from acceptance" in evidence
    assert "1680s wall clock" in evidence
    assert S_ACCEPT.isoformat() in evidence
    # The lane is named, but never accused.
    assert SIBLING_STALE_SHA[:12] in evidence
    assert "which is behind relative to" not in evidence


def test_the_8d52a7ccf_timeline_converges_when_the_job_can_afford_the_budget() -> None:
    """The same timeline with a wall clock that covers the lane's budget.

    This is what the fix buys once OMN-18637's ceiling work lands: the queue is
    no longer charged to the lane, so a job able to watch for the whole budget
    reports the PASS the lane earned.
    """
    clock = _Clock(S_GUARD_START)
    result = _sibling_wait(
        clock=clock,
        converges_at=S_RECREATE,
        acceptance_at=S_ACCEPT,
        wall_clock=timedelta(minutes=40),
    )
    assert result.outcome is EnumConvergenceOutcome.OK
    assert _sibling_evidence(result).startswith("lane vendors omnimarket at")


# --- fail-closed cases the port does not relax -----------------------------


def test_an_unreadable_lane_with_a_spent_budget_still_fails_closed() -> None:
    clock = _Clock(S_GUARD_START)
    result = _sibling_wait(
        clock=clock,
        converges_at=None,
        acceptance_at=S_ACCEPT,
        wall_clock=timedelta(minutes=40),
        unreadable_until=S_GUARD_START + timedelta(hours=4),
    )
    assert result.outcome is EnumConvergenceOutcome.FAIL
    evidence = _sibling_evidence(result)
    assert "could not be read" in evidence
    assert "fails closed" in evidence


def test_an_unrecognised_compare_status_is_refused_not_waited_out() -> None:
    clock = _Clock(S_GUARD_START)

    def observe() -> ModelSiblingObservation:
        return ModelSiblingObservation(
            lane_revision=SIBLING_STALE_SHA,
            containment="unknown_status",
            unreadable_reason="",
        )

    result = run_sibling_convergence_wait(
        observe=observe,
        resolve_acceptance=lambda: ModelAcceptanceProbe(
            acceptance=ModelAgentAcceptance(
                correlation_id=CORRELATION_ID,
                accepted_at=S_ACCEPT,
                source="http://host.docker.internal:8098",
            ),
            reason="",
        ),
        declared_budget=DECLARED,
        wall_clock=WALL_CLOCK,
        poll_interval=timedelta(seconds=60),
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.outcome is EnumConvergenceOutcome.FAIL
    assert "does not recognise" in _sibling_evidence(result)


def test_an_observation_cannot_be_both_readable_and_unreadable() -> None:
    with pytest.raises(ValueError, match="EXACTLY one"):
        ModelSiblingObservation(
            lane_revision=SIBLING_MERGE_SHA, containment="ahead", unreadable_reason="x"
        )
    with pytest.raises(ValueError, match="EXACTLY one"):
        ModelSiblingObservation(lane_revision="", containment="", unreadable_reason="")


def test_the_sibling_guard_exits_three_for_indeterminate() -> None:
    """Non-zero, and distinct from 1, so a caller reading only the status can
    still tell 'the lane did not converge' from 'nothing was established'."""
    assert SIBLING_EXIT_INDETERMINATE == 3


# ---------------------------------------------------------------------------
# OMN-18685 -- the sibling REUSABLE's own wiring.
#
# The guard can only start the budget at acceptance if the workflow hands it a
# correlation id and an agent surface, so the flags are asserted here for the
# same reason the direct caller's are above: a capability the calling YAML never
# reaches is detection, not enforcement. This is the file omnimarket,
# omnibase_core and omnibase_compat all call, at a pinned ref.
# ---------------------------------------------------------------------------
SIBLING_WORKFLOW = REPO_ROOT / ".github/workflows/runtime-rebuild-trigger-reusable.yml"


def _sibling_job() -> dict:
    import yaml

    model = yaml.safe_load(SIBLING_WORKFLOW.read_text(encoding="utf-8"))
    job = model["jobs"]["verify-sibling-converged"]
    assert isinstance(job, dict)
    return job


def _sibling_converge_step() -> dict:
    step = next(s for s in _sibling_job()["steps"] if s.get("id") == "converge")
    assert isinstance(step, dict)
    return step


def test_the_sibling_converge_step_is_handed_the_agent_surface_and_correlation() -> (
    None
):
    body = _sibling_converge_step()["run"]
    assert "--agent-url" in body
    assert "--correlation-id" in body
    assert "--wall-clock-seconds" in body
    # The lane's budget is unchanged: the flags above bound the WAIT, and a fix
    # that quietly widened the grant would show up right here.
    assert "--wait-timeout 25m" in body


def test_the_sibling_converge_wall_clock_is_derived_not_typed() -> None:
    body = _sibling_converge_step()["run"]
    assert "lane_settle_budget.py" in body
    assert "--converge-wall-clock" in body


def test_the_sibling_converge_step_survives_an_indeterminate_exit() -> None:
    """Exit 3 must not abort the job before the receipt is written.

    A job that dies on the guard's non-zero status uploads no receipt at all,
    which is the one outcome worse than a failing one -- and it is the shape
    that makes "it failed" and "nobody ran it" indistinguishable.
    """
    assert _sibling_converge_step()["continue-on-error"] is True


def test_the_sibling_emit_step_reads_the_three_valued_verdict() -> None:
    emit = next(
        s for s in _sibling_job()["steps"] if "emit" in str(s.get("name", "")).lower()
    )
    assert "steps.converge.outputs.verdict" in emit["env"]["CONVERGE_VERDICT"]
    assert "steps.converge.outputs.evidence" in emit["env"]["CONVERGE_EVIDENCE"]
    assert "indeterminate" in emit["run"]
    # The evidence is dereferenced from env in the shell, never interpolated
    # into the run body (OMN-18638, the workflow script-injection rule).
    assert "${{" not in emit["run"]


def test_the_sibling_job_ceiling_and_settle_budget_are_unchanged() -> None:
    """This port moves the ANCHOR, never the grant."""
    assert _sibling_job()["timeout-minutes"] == 45


def test_the_sibling_delivery_announcement_is_still_gated_on_the_verify_job() -> None:
    """AC: the fail-closed direction is untouched by `continue-on-error`.

    `continue-on-error` is on the converge STEP. The emit step still exits
    non-zero on any non-PASS receipt, so the JOB still fails, so this job's
    default `success()` gating still skips the staging announcement. Nothing in
    this change lets a non-converged sibling announce itself.
    """
    import yaml

    model = yaml.safe_load(SIBLING_WORKFLOW.read_text(encoding="utf-8"))
    deliver = model["jobs"]["deliver-sibling-candidate"]
    assert deliver["needs"] == ["trigger-rebuild", "verify-sibling-converged"]
    assert "if" not in deliver, (
        "an `if` here would replace the implicit success() gating that keeps a "
        "failed lab pass from announcing itself"
    )
    emit = next(
        s for s in _sibling_job()["steps"] if "emit" in str(s.get("name", "")).lower()
    )
    assert emit["if"] == "always()"
