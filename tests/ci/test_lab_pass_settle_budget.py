# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18436 -- the compose-dev settle budget is declared, not a job remainder.

THE INCIDENT, in the job's own arithmetic. ``verify-lane-converged`` declared
three constants that could not all hold -- a job ceiling of 1800 s
(``timeout-minutes``), a convergence allowance of 1500 s (``--wait-timeout``)
and a reserved tail of 120 s -- and computed the settle budget in shell as
``ceiling - elapsed - tail``. A
convergence that used its full allowance therefore left **180 s** for a lane
boot measured at **463-668 s** on four separate emitted receipts. The receipt
FAILED on timing alone -- healthy lane, correct sha, convergence proven -- and
rule 24(b) fails closed on a FAIL receipt, so that receipt then refused the sha
for staging delivery. One of the seven compose-dev receipts retained on
2026-09-16 (``6e41f3e0``) failed exactly that way on ``ready_effects``.

THE SECOND DEFECT, in the same job. The probe step runs under ``if: always()``,
so when convergence FAILS the four HTTP checks still run -- against the PREVIOUS
lane generation, still up and still serving. Receipt ``4853e0e1`` is that shape
verbatim: ``deployed_revision`` FAIL with ``ready_effects`` TRUE. Two true
statements about two different containers, and nothing in the receipt saying so.

THE THIRD, which is why the first two were expensive rather than merely wrong.
A budget-exhausted boot and an unhealthy lane both arrived as
``ready_effects: fail``. Telling those apart is the whole job of a receipt.

These tests pin the arithmetic, the bounds it is derived from, and the SHAPE of
the reasoning -- so an edit that drops the ceiling back, or lowers the
declaration under the measurement, has to confront the numbers rather than a
comment.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.lab_pass_receipt import (
    GENERATION_CHECK,
    SETTLE_BUDGET_CHECK,
    SETTLE_TIMEOUT_CHECK,
    ModelLaneGeneration,
    ModelSettleBudget,
    ModelSettleOutcome,
    generation_check,
    parse_generation,
    probe_compose_dev,
    settle_budget_check,
    settle_timeout_check,
)
from scripts.ci.lane_settle_budget import (
    DEFAULT_DECLARATION_PATH,
    SettleBudgetError,
    affordable_seconds,
    assert_declaration_within_bounds,
    derive_settle_budget,
    load_declaration,
    max_declared_start_period_seconds,
    parse_compose_duration,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LANE = "compose-dev"

WORKFLOWS = (
    REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml",
    REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger-reusable.yml",
)

#: The reserved tail both workflows declare, held back so the receipt can still
#: be built, validated and uploaded after the probe returns.
RESERVED_TAIL_SECONDS = 120

#: The ceiling the job carried when the defect was live. Kept as a NUMBER rather
#: than as prose so the reproduction below is arithmetic a reader can check.
CEILING_AT_THE_INCIDENT_SECONDS = 30 * 60

#: The slowest boot any emitted compose-dev receipt has recorded
#: (``732fd291``: "lane ready after 668s of an 837s settle budget"). A FLOOR on
#: the real cost, never an estimate of it.
WORST_OBSERVED_BOOT_SECONDS = 668


def _fake_clock(monkeypatch: Any) -> None:
    """A virtual clock, so a settle-budget test costs no wall-clock time.

    The production loop measures with ``time.monotonic``; patching only
    ``time.sleep`` would leave it spinning against a real deadline, which is how
    a unit test quietly becomes a fifteen-minute one.
    """
    now = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: now["t"])
    monkeypatch.setattr(
        "time.sleep", lambda seconds: now.__setitem__("t", now["t"] + seconds)
    )


def _write_declaration(tmp_path: Path, **overrides: Any) -> Path:
    entry: dict[str, Any] = {
        "settle_budget_seconds": 900,
        "compose_file": "docker/docker-compose.infra.yml",
        "readiness_services": ["omninode-runtime", "runtime-effects", "projection-api"],
        "observed_boot_seconds": [{"seconds": 668, "sha": "732fd291"}],
    }
    entry.update(overrides)
    path = tmp_path / "declaration.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "lanes": {LANE: entry}}), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# The reproduction: the job's own three numbers were unsatisfiable
# ---------------------------------------------------------------------------
class TestTheIncidentIsArithmeticNotBadLuck:
    def test_the_old_ceiling_could_not_afford_the_lane_s_measured_boot(self) -> None:
        """The failing window was structural: a 5-minute band of convergence
        times in which the receipt could not pass whatever the lane did."""
        converge_seconds = _converge_timeout_seconds(WORKFLOWS[0])
        worst_case_affordable = affordable_seconds(
            CEILING_AT_THE_INCIDENT_SECONDS, converge_seconds, RESERVED_TAIL_SECONDS
        )
        assert worst_case_affordable < WORST_OBSERVED_BOOT_SECONDS, (
            "this test encodes the incident; if the old ceiling could in fact "
            "afford the measured boot then the diagnosis was wrong"
        )
        # And the fix is exactly that this can no longer happen.
        budget = derive_settle_budget(
            lane=LANE,
            job_ceiling_seconds=_job_ceiling_seconds(WORKFLOWS[0]),
            elapsed_seconds=converge_seconds,
            reserved_tail_seconds=RESERVED_TAIL_SECONDS,
        )
        assert budget.sufficient, (
            f"a convergence using its full {converge_seconds}s allowance still "
            f"leaves only {budget.affordable_seconds}s against a declared "
            f"{budget.declared_seconds}s -- the defect is not closed"
        )
        assert budget.granted_seconds == budget.declared_seconds


# ---------------------------------------------------------------------------
# AC1 -- declared, fail-closed, and not a function of elapsed time
# ---------------------------------------------------------------------------
class TestTheBudgetIsDeclaredAndFailsClosed:
    def test_the_budget_does_not_move_when_the_job_s_elapsed_time_moves(self) -> None:
        """The lane's boot time is a property of the lane, not of how long the
        deploy-agent queue happened to be."""
        quick = derive_settle_budget(LANE, 2700, 10, RESERVED_TAIL_SECONDS)
        slow = derive_settle_budget(LANE, 2700, 1400, RESERVED_TAIL_SECONDS)
        assert quick.declared_seconds == slow.declared_seconds
        assert quick.granted_seconds == slow.granted_seconds
        # The affordance still moves -- it is a real fact about the run -- and
        # that is precisely what the sufficiency check reports.
        assert quick.affordable_seconds != slow.affordable_seconds

    def test_a_missing_declaration_file_raises_rather_than_returning_a_remainder(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(SettleBudgetError, match="unreadable"):
            derive_settle_budget(
                LANE, 2700, 10, RESERVED_TAIL_SECONDS, path=tmp_path / "absent.yaml"
            )

    def test_an_undeclared_lane_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "declaration.yaml"
        path.write_text(
            yaml.safe_dump({"schema_version": 1, "lanes": {"other": {}}}),
            encoding="utf-8",
        )
        with pytest.raises(SettleBudgetError, match="declares no settle budget"):
            load_declaration(LANE, path)

    def test_a_declaration_from_another_contract_version_raises(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "declaration.yaml"
        path.write_text(
            yaml.safe_dump({"schema_version": 99, "lanes": {LANE: {}}}),
            encoding="utf-8",
        )
        with pytest.raises(SettleBudgetError, match="schema_version"):
            load_declaration(LANE, path)

    @pytest.mark.parametrize("bad", [0, -1, "900", 900.5, True])
    def test_a_non_positive_or_non_integer_budget_raises(
        self, tmp_path: Path, bad: Any
    ) -> None:
        with pytest.raises(SettleBudgetError, match="positive integer"):
            load_declaration(
                LANE, _write_declaration(tmp_path, settle_budget_seconds=bad)
            )

    def test_a_budget_with_no_observation_behind_it_raises(
        self, tmp_path: Path
    ) -> None:
        """A number nobody measured has nothing to keep it honest."""
        with pytest.raises(SettleBudgetError, match="non-empty list"):
            load_declaration(
                LANE, _write_declaration(tmp_path, observed_boot_seconds=[])
            )

    def test_the_committed_declaration_loads(self) -> None:
        """Positive control on every refusal above: the real file is readable,
        so the raises are about the input and not about the reader."""
        declaration = load_declaration(LANE)
        assert declaration.settle_budget_seconds > 0
        assert declaration.observed_boot_seconds
        assert DEFAULT_DECLARATION_PATH.exists()


# ---------------------------------------------------------------------------
# AC2 -- both bounds, read from parsed models rather than from prose
# ---------------------------------------------------------------------------
class TestTheDeclarationIsBoundedByTheLaneSOwnModel:
    def test_the_committed_declaration_sits_inside_both_bounds(self) -> None:
        declaration = load_declaration(LANE)
        assert_declaration_within_bounds(declaration)
        upper = max_declared_start_period_seconds(declaration)
        assert declaration.settle_budget_seconds <= upper
        assert declaration.settle_budget_seconds >= declaration.worst_observed_seconds

    def test_the_upper_bound_is_the_real_compose_start_period(self) -> None:
        """Parsed from docker/docker-compose.infra.yml, not transcribed."""
        declaration = load_declaration(LANE)
        model = yaml.safe_load(
            (REPO_ROOT / declaration.compose_file).read_text(encoding="utf-8")
        )
        expected = max(
            parse_compose_duration(
                model["services"][name]["healthcheck"]["start_period"]
            )
            for name in declaration.readiness_services
        )
        assert max_declared_start_period_seconds(declaration) == expected

    def test_a_budget_above_the_lane_s_start_period_raises(
        self, tmp_path: Path
    ) -> None:
        declaration = load_declaration(
            LANE, _write_declaration(tmp_path, settle_budget_seconds=99_999)
        )
        with pytest.raises(SettleBudgetError, match="exceeds"):
            assert_declaration_within_bounds(declaration)

    def test_a_budget_below_the_worst_observed_boot_raises(
        self, tmp_path: Path
    ) -> None:
        """The exact regression: a budget under the measurement fails on timing
        alone, which is the defect and not a tighter bound."""
        declaration = load_declaration(
            LANE,
            _write_declaration(
                tmp_path,
                settle_budget_seconds=300,
                observed_boot_seconds=[{"seconds": 668, "sha": "732fd291"}],
            ),
        )
        with pytest.raises(SettleBudgetError, match="worst boot"):
            assert_declaration_within_bounds(declaration)

    def test_a_readiness_service_with_no_start_period_raises(
        self, tmp_path: Path
    ) -> None:
        """Skipping it would silently LOWER the ceiling the bound compares
        against, which is the direction that lets a too-large budget through."""
        declaration = load_declaration(
            LANE,
            _write_declaration(tmp_path, readiness_services=["postgres", "phoenix"]),
        )
        assert max_declared_start_period_seconds(declaration) > 0  # positive control
        declaration = load_declaration(
            LANE, _write_declaration(tmp_path, readiness_services=["migration-runner"])
        )
        with pytest.raises(SettleBudgetError):
            max_declared_start_period_seconds(declaration)

    @pytest.mark.parametrize(
        ("raw", "seconds"),
        [("1800s", 1800.0), ("2m", 120.0), ("1h", 3600.0), ("500ms", 0.5), (90, 90.0)],
    )
    def test_compose_durations_parse(self, raw: Any, seconds: float) -> None:
        assert parse_compose_duration(raw) == seconds

    def test_an_unrecognised_duration_suffix_raises(self) -> None:
        """Coercing `1800d` to 1800 would be wrong by five orders of magnitude
        and would read as a correct number."""
        with pytest.raises(SettleBudgetError, match="not a compose duration"):
            parse_compose_duration("1800d")


# ---------------------------------------------------------------------------
# AC3 -- the job ceiling is DERIVED from the declared parts
# ---------------------------------------------------------------------------
def _workflow(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _verify_job(path: Path) -> dict[str, Any]:
    jobs = _workflow(path)["jobs"]
    for name, job in jobs.items():
        if "verify" in name and "converged" in name:
            return dict(job)
    msg = f"{path.name} declares no verify-converged job"
    raise AssertionError(msg)


def _job_ceiling_seconds(path: Path) -> int:
    return int(_verify_job(path)["timeout-minutes"]) * 60


def _converge_timeout_seconds(path: Path) -> int:
    """The convergence allowance, read from the step's own argument."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"--wait-timeout\s+(\d+)m", text)
    assert match is not None, f"{path.name} declares no convergence --wait-timeout"
    return int(match.group(1)) * 60


class TestTheJobCeilingIsDerivedFromTheDeclaredParts:
    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_the_ceiling_covers_convergence_plus_the_declared_settle_plus_the_tail(
        self, path: Path
    ) -> None:
        declared = load_declaration(LANE).settle_budget_seconds
        required = _converge_timeout_seconds(path) + declared + RESERVED_TAIL_SECONDS
        ceiling = _job_ceiling_seconds(path)
        assert ceiling >= required, (
            f"{path.name}: ceiling {ceiling}s cannot contain its own declared "
            f"parts ({_converge_timeout_seconds(path)}s convergence + "
            f"{declared}s settle + {RESERVED_TAIL_SECONDS}s tail = {required}s). "
            "That is the OMN-18436 defect: the receipt then FAILs on timing "
            "alone and rule 24(b) refuses the sha for staging."
        )

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_the_ceiling_is_still_bounded(self, path: Path) -> None:
        """This job holds the single omnibase-deploy runner slot. An unbounded
        ceiling would sit on a wedged lane instead of failing visibly."""
        assert _job_ceiling_seconds(path) <= 60 * 60

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_the_probe_step_passes_the_ceiling_and_never_a_remainder(
        self, path: Path
    ) -> None:
        text = path.read_text(encoding="utf-8")
        assert "scripts/ci/lane_settle_budget.py" in text
        assert "--job-ceiling-seconds" in text
        assert "--elapsed-seconds" in text
        assert "--reserved-tail-seconds" in text
        assert "--settle-budget-json" in text
        assert "--settle-timeout-seconds" not in text, (
            "passing a precomputed remainder is the path this ticket removed; "
            "the budget is derived from the declaration instead"
        )
        assert "SETTLE=$((" not in text, (
            "the budget arithmetic must not live in a run: block, where nothing "
            "tests it"
        )

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_the_env_ceiling_matches_the_job_ceiling(self, path: Path) -> None:
        """The env value the probe step passes must be the job's real ceiling;
        a drifted copy would report an affordance the job does not have."""
        job = _verify_job(path)
        probe = next(
            step
            for step in job["steps"]
            if str(step.get("id", "")) == "probe"
            or "Probe" in str(step.get("name", ""))
        )
        assert int(probe["env"]["JOB_TIMEOUT_MINUTES"]) == int(job["timeout-minutes"])
        assert int(probe["env"]["RESERVED_TAIL_SECONDS"]) == RESERVED_TAIL_SECONDS

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_no_force_or_skip_was_added_to_the_probe(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        assert "--force" not in text
        assert "--skip-settle" not in text


# ---------------------------------------------------------------------------
# AC4 -- a short grant is its own named check, with both numbers
# ---------------------------------------------------------------------------
class TestAnUnaffordableBudgetIsNamedRatherThanImplied:
    def test_an_insufficient_grant_fails_and_names_both_numbers(self) -> None:
        budget = derive_settle_budget(
            LANE, CEILING_AT_THE_INCIDENT_SECONDS, 1500, RESERVED_TAIL_SECONDS
        )
        check = settle_budget_check(budget)
        assert check.name == SETTLE_BUDGET_CHECK
        assert check.ok is False
        assert str(budget.declared_seconds) in check.evidence
        assert str(budget.affordable_seconds) in check.evidence
        assert "run out of CLOCK" in check.evidence

    def test_a_sufficient_grant_passes_and_still_carries_its_numbers(self) -> None:
        """Emitted on BOTH verdicts: a check that appeared only on failure is
        indistinguishable from one that was never run."""
        budget = derive_settle_budget(LANE, 2700, 1500, RESERVED_TAIL_SECONDS)
        check = settle_budget_check(budget)
        assert check.ok is True
        assert str(budget.declared_seconds) in check.evidence
        assert str(budget.affordable_seconds) in check.evidence

    def test_the_grant_never_exceeds_what_the_job_can_write_a_receipt_within(
        self,
    ) -> None:
        budget = derive_settle_budget(
            LANE, CEILING_AT_THE_INCIDENT_SECONDS, 1500, RESERVED_TAIL_SECONDS
        )
        assert budget.granted_seconds == budget.affordable_seconds
        assert budget.granted_seconds < budget.declared_seconds


# ---------------------------------------------------------------------------
# AC5 -- a budget-exhausted boot is its own check, with the measured wait
# ---------------------------------------------------------------------------
class TestTimingOutIsDistinctFromBeingUnhealthy:
    def test_an_expired_wait_reports_the_measured_seconds_and_what_was_pending(
        self,
    ) -> None:
        outcome = ModelSettleOutcome(
            ready=False,
            waited_seconds=180.0,
            granted_seconds=180.0,
            pending=("http://lane:8086/ready",),
        )
        check = settle_timeout_check(outcome)
        assert check.name == SETTLE_TIMEOUT_CHECK
        assert check.ok is False
        assert "180s" in check.evidence
        assert "8086" in check.evidence

    def test_a_lane_that_came_up_passes_and_records_its_boot_time(self) -> None:
        check = settle_timeout_check(
            ModelSettleOutcome(
                ready=True, waited_seconds=488.0, granted_seconds=900.0, pending=()
            )
        )
        assert check.ok is True
        assert "488s" in check.evidence

    def test_the_probe_emits_it_when_a_budget_was_granted(
        self, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._http_get",
            lambda url, timeout: (0, "URLError: [Errno 111] Connection refused"),
        )
        _fake_clock(monkeypatch)
        checks = probe_compose_dev(
            main_url="http://lane:8085",
            effects_url="http://lane:8086",
            timeout_seconds=1.0,
            settle_timeout_seconds=180.0,
            projection_url="http://lane:3002",
        )
        timeout = next(c for c in checks if c.name == SETTLE_TIMEOUT_CHECK)
        assert timeout.ok is False
        assert "timed out before ready" in timeout.evidence

    def test_an_ad_hoc_read_with_no_budget_makes_no_such_claim(
        self, monkeypatch: Any
    ) -> None:
        """Positive control on the emission rule: the check states whether the
        lane came up inside a GRANT, which is meaningless without one."""
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._http_get",
            lambda url, timeout: (0, "URLError: [Errno 111] Connection refused"),
        )
        checks = probe_compose_dev(
            main_url="http://lane:8085",
            effects_url="http://lane:8086",
            timeout_seconds=1.0,
            settle_timeout_seconds=0.0,
            projection_url="http://lane:3002",
        )
        assert SETTLE_TIMEOUT_CHECK not in {c.name for c in checks}
        assert SETTLE_BUDGET_CHECK not in {c.name for c in checks}
        assert GENERATION_CHECK not in {c.name for c in checks}


# ---------------------------------------------------------------------------
# AC6 -- the reads are bound to the generation convergence observed
# ---------------------------------------------------------------------------
def _generation(
    container: str = "omninode-runtime", **over: str
) -> ModelLaneGeneration:
    payload = {
        "Id": over.get("container_id", "a" * 64),
        "Name": f"/{container}",
        "Image": over.get("image", "sha256:" + "b" * 64),
        "Config": {"Labels": {"org.opencontainers.image.revision": "c" * 40}},
    }
    return parse_generation(payload)


class TestTheProbeIsBoundToTheGenerationConvergenceRead:
    def test_a_matching_pair_passes(self) -> None:
        expected = _generation()
        check = generation_check(expected, parse_generation(expected.to_json()))
        assert check.name == GENERATION_CHECK
        assert check.ok is True
        assert expected.container_id[:12] in check.evidence

    def test_a_recreate_between_convergence_and_the_probe_fails_and_names_both(
        self,
    ) -> None:
        expected = _generation()
        observed = _generation(container_id="d" * 64)
        check = generation_check(expected, observed)
        assert check.ok is False
        assert expected.container_id[:12] in check.evidence
        assert observed.container_id[:12] in check.evidence

    def test_a_new_image_on_the_same_container_id_is_still_a_new_generation(
        self,
    ) -> None:
        check = generation_check(_generation(), _generation(image="sha256:" + "e" * 64))
        assert check.ok is False

    def test_no_generation_record_fails_naming_the_absence(self) -> None:
        """The receipt-4853e0e1 shape: convergence failed, the previous lane
        answered, and the greens were about a container nobody named."""
        check = generation_check(None, _generation())
        assert check.ok is False
        assert "published no container generation" in check.evidence

    def test_an_unreadable_container_at_probe_time_fails(self) -> None:
        check = generation_check(_generation(), None, "RuntimeError: no such container")
        assert check.ok is False
        assert "no such container" in check.evidence

    def test_the_probe_emits_the_check_only_when_the_caller_claims_a_binding(
        self, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._http_get",
            lambda url, timeout: (0, "URLError: [Errno 111] Connection refused"),
        )
        expected = _generation()
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt.read_lane_generation",
            lambda container: expected,
        )
        bound = probe_compose_dev(
            main_url="http://lane:8085",
            effects_url="http://lane:8086",
            timeout_seconds=1.0,
            settle_timeout_seconds=0.0,
            projection_url="http://lane:3002",
            expected_generation=expected,
            generation_container="omninode-runtime",
        )
        assert next(c for c in bound if c.name == GENERATION_CHECK).ok is True

    def test_a_generation_record_survives_the_github_output_boundary(self) -> None:
        """It crosses as one line of JSON; a newline would truncate it and the
        probe would read a binding claim it cannot parse."""
        rendered = _generation().to_json()
        assert "\n" not in rendered
        assert json.loads(rendered)["container_id"]

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_the_workflow_passes_the_convergence_generation_through(
        self, path: Path
    ) -> None:
        text = path.read_text(encoding="utf-8")
        assert "--generation-container" in text
        assert "--expect-generation" in text
        assert "steps.converge.outputs.generation" in text

    def test_both_convergence_guards_publish_a_generation(self) -> None:
        """Either guard may be the one that ran; a guard that published nothing
        would fail the binding check on every run of its own workflow."""
        for module in ("check_dev_lane_staleness.py", "check_lane_sibling_revision.py"):
            text = (REPO_ROOT / "scripts" / "ci" / module).read_text(encoding="utf-8")
            assert "read_lane_generation" in text, module
            assert "generation=" in text or '"generation"' in text, module
