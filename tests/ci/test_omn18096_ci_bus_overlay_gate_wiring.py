# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18096: the CI-bus overlay binding gate must be able to block a merge.

CLAUDE.md Operating Rule 5 — a check that is not a merge condition is advisory
and gets ignored. ``ci-bus-overlay-binding.yml`` landed in #3371 (OMN-18060,
``cc897440``) deliberately unregistered, with that residual named in its own
header. This file is the other half: the registration, and the producer-side
properties that make the registration meaningful.

On this repo the wiring is a two-part claim and the first part is a trap. Live
readback::

    gh api repos/OmniNode-ai/omnibase_infra/branches/dev/protection/required_status_checks \\
        --jq '.contexts'
    ["CI Summary"]

``dev`` requires exactly ONE context (the OMN-4497 single-umbrella design), so
:data:`EXPECTED_EXTERNAL_CONTEXTS` — the tuple ``CI Summary`` asserts
present-completed-success on the PR head — IS the external enforcement surface.
Membership is what makes this gate blocking; there is no branch-protection
signal that would reveal its loss.

WHAT THE GATE COVERS, AND WHICH DIRECTION. ``config/ci_bus_lanes.yaml`` is
written in omnimarket and validated here by ``ModelCiBusOverlay``
(``extra="forbid"``). The per-PR binding test
(``tests/unit/scripts/test_ci_bus_overlay_live_binding.py``) covers the
CONSUMER direction: a change to the model here that no longer accepts the live
config reds on this repo's own PR. The workflow's schedule covers the PRODUCER
direction: a key added in omnimarket creates no pull request here at all, and
before this gate existed it surfaced hours later as a red rebuild-trigger run
on somebody else's merge (OMN-18012 on 2026-09-07, OMN-18060 on 2026-09-09).

FAIL, NEVER SKIP. CLAUDE.md rule 16 — a zero result is not evidence of absence.
An empty sparse checkout is the one failure shape that would turn this job into
a green no-op, so the workflow refuses it before pytest runs, and no step may
carry ``continue-on-error`` or swallow a non-zero exit with ``|| true``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    EXPECTED_EXTERNAL_CONTEXTS,
    EXTERNAL_GOOD_CONCLUSIONS,
    MEASURED_NOT_ENFORCED_CONTEXTS,
    POST_FIXTURE_WINDOW_CONTEXTS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    evaluate,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: The check-run name GitHub publishes for this job, read back live from
#: #3371's head ``ac67958009fffd76f3782aad1f70e085ab07283d`` and from the merge
#: commit ``cc8974408d6c543311a40b8beb414787a80fef2e``.
GATE_CONTEXT = "ci-bus-overlay-binding"
GATE_WORKFLOW = "ci-bus-overlay-binding.yml"


def _workflow(name: str) -> dict[Any, Any]:
    # dict[Any, Any], not dict[str, Any]: PyYAML resolves the bare `on:` key to
    # the boolean True, so the mapping is genuinely not str-keyed.
    loaded: dict[Any, Any] = yaml.safe_load(
        (WORKFLOWS / name).read_text(encoding="utf-8")
    )
    return loaded


def _triggers(workflow: dict[Any, Any]) -> set[str]:
    raw = workflow.get(True, workflow.get("on"))
    if isinstance(raw, dict):
        return set(raw)
    if isinstance(raw, list):
        return set(raw)
    return {str(raw)}


def _steps() -> list[dict[str, Any]]:
    return list(_workflow(GATE_WORKFLOW)["jobs"][GATE_CONTEXT]["steps"])


def _all_gates_success() -> list[dict[str, Any]]:
    """A full, passing in-run snapshot: every strict AND skippable gate green."""
    return [
        {"name": name, "status": "completed", "conclusion": "success"}
        for name in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)
    ]


def _complete_external_payload() -> list[dict[str, Any]]:
    """A synthetic head payload in which every asserted context is green."""
    return [
        {"name": context, "status": "completed", "conclusion": "success"}
        for context in EXPECTED_EXTERNAL_CONTEXTS
    ]


class TestTheGateCanBlockAMerge:
    def test_the_context_is_asserted_by_ci_summary(self) -> None:
        """Membership in the tuple is the ONLY thing making this block.

        Dropping it does not downgrade the gate to a weaker surface; it returns
        the gate to the advisory state #3371 shipped it in, with nothing in
        branch protection to reveal the loss.
        """
        assert GATE_CONTEXT in EXPECTED_EXTERNAL_CONTEXTS

    def test_the_context_is_not_also_declared_unenforced(self) -> None:
        assert GATE_CONTEXT not in MEASURED_NOT_ENFORCED_CONTEXTS

    def test_the_producer_workflow_exists(self) -> None:
        assert (WORKFLOWS / GATE_WORKFLOW).is_file()

    def test_the_job_id_renders_the_asserted_context_name(self) -> None:
        """The check-run name CI Summary keys on is the job name; pin them equal."""
        workflow = _workflow(GATE_WORKFLOW)
        job = workflow["jobs"].get(GATE_CONTEXT)
        assert job is not None, (
            f"{GATE_WORKFLOW} has no job id {GATE_CONTEXT!r}; the check-run name "
            "the umbrella asserts would silently stop existing."
        )
        assert job.get("name") == GATE_CONTEXT

    def test_the_producer_reports_on_pr_and_merge_group(self) -> None:
        triggers = _triggers(_workflow(GATE_WORKFLOW))
        assert "pull_request" in triggers
        assert "merge_group" in triggers, (
            "an asserted context that never reports on a queue SHA wedges every "
            "merge should a queue be re-enabled on this repo."
        )

    def test_the_producer_keeps_its_standing_schedule(self) -> None:
        """The schedule is the half that covers the producer-side direction.

        A key added in omnimarket creates no pull request here, so a PR-only
        gate still makes an unrelated merge pay for it. Registering the context
        does not replace the schedule and must not become a reason to drop it.
        """
        assert "schedule" in _triggers(_workflow(GATE_WORKFLOW))

    def test_the_producer_has_no_skip_path(self) -> None:
        """No ``needs:``, no job-level ``if:``, no path filter.

        A skipped producer is the OMN-15057 vector-5 silent pass. It is defused
        twice over (``EXTERNAL_GOOD_CONCLUSIONS`` admits only ``success``), but
        the cheapest place to keep it defused is the producer.
        """
        workflow = _workflow(GATE_WORKFLOW)
        job = workflow["jobs"][GATE_CONTEXT]
        assert "needs" not in job
        assert "if" not in job

        raw_on = workflow.get(True, workflow.get("on"))
        assert isinstance(raw_on, dict)
        for event, spec in raw_on.items():
            if not isinstance(spec, dict):
                continue
            assert "paths" not in spec and "paths-ignore" not in spec, (
                f"{GATE_WORKFLOW} path-filters {event}; a path-filtered required "
                "context never reports on PRs that miss its paths, which is the "
                "never-reports wedge recorded in MEASURED_NOT_ENFORCED_CONTEXTS."
            )

    def test_skipped_is_not_a_good_external_conclusion(self) -> None:
        assert frozenset({"success"}) == EXTERNAL_GOOD_CONCLUSIONS


class TestTheGateFailsRatherThanSkips:
    """CLAUDE.md rule 16 — an unreachable producer must red, never no-op green."""

    def test_no_step_is_continue_on_error(self) -> None:
        offenders = [
            str(step.get("name", step.get("uses", "?")))
            for step in _steps()
            if step.get("continue-on-error") is True
        ]
        assert offenders == [], (
            "a continue-on-error step turns an unreachable omnimarket, a failed "
            f"checkout or a red validation into a green run: {offenders}"
        )

    def test_no_step_swallows_a_non_zero_exit(self) -> None:
        for step in _steps():
            run = str(step.get("run", ""))
            assert "|| true" not in run, (
                f"step {step.get('name')!r} swallows a non-zero exit; the whole "
                "point of this gate is that the failure is visible here"
            )

    def test_the_cross_repo_checkout_is_not_softened(self) -> None:
        """The fetch itself is the failure mode the ticket names first.

        An unresolvable head or an unreadable file must fail the job. Both
        surface as a failing ``actions/checkout`` step, which is only true while
        that step neither continues on error nor is conditioned away.
        """
        checkouts = [
            step
            for step in _steps()
            if str(step.get("uses", "")).startswith("actions/checkout")
            and step.get("with", {}).get("repository") == "OmniNode-ai/omnimarket"
        ]
        assert len(checkouts) == 1, "expected exactly one omnimarket checkout step"
        assert checkouts[0].get("continue-on-error") is not True
        assert "if" not in checkouts[0]

    def test_an_empty_sparse_checkout_is_refused_before_pytest(self) -> None:
        """A silently-empty checkout would make every case skip and the job green.

        The workflow refuses it in shell, ahead of the pytest step, because the
        pytest step's own guard is a ``skip`` when the variable is unset — and a
        skip reads as a pass in a summary line.
        """
        guard = next(
            (
                s
                for s in _steps()
                if "Assert the overlay checkout" in str(s.get("name"))
            ),
            None,
        )
        assert guard is not None, (
            "the overlay-presence guard is gone; without it a sparse checkout "
            "that produced nothing yields a green run that proves nothing."
        )
        run = str(guard["run"])
        assert "set -euo pipefail" in run
        assert "exit 1" in run

        names = [str(s.get("name", "")) for s in _steps()]
        guard_at = names.index(str(guard["name"]))
        validate_at = next(
            i for i, n in enumerate(names) if "Validate the live overlay" in n
        )
        assert guard_at < validate_at, "the guard must run before the validation step"

    def test_the_validation_step_exports_the_live_path(self) -> None:
        """Without the env var the bound cases skip and the job is vacuously green."""
        validate = next(
            s for s in _steps() if "Validate the live overlay" in str(s.get("name", ""))
        )
        assert validate["env"]["CI_BUS_OVERLAY_LIVE_PATH"]
        assert "tests/unit/scripts/test_ci_bus_overlay_live_binding.py" in str(
            validate["run"]
        )


class TestTheRegistrationIsLoadBearingAtRuntime:
    """Membership is asserted against a live payload, not merely declared.

    ``POST_FIXTURE_WINDOW_CONTEXTS`` excludes this name from the historical
    fixture replays — the workflow did not exist when those windows were
    captured — so these cases are what prove the exclusion changed nothing about
    the live verdict.
    """

    def test_the_context_is_excluded_only_from_historical_replays(self) -> None:
        assert GATE_CONTEXT in POST_FIXTURE_WINDOW_CONTEXTS
        assert GATE_CONTEXT in EXPECTED_EXTERNAL_CONTEXTS

    def test_a_complete_payload_greens(self) -> None:
        """Guard the premise of the next two cases: the payload is otherwise clean."""
        code, report = evaluate(
            _all_gates_success(),
            check_runs=_complete_external_payload(),
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_SUCCESS, report

    def test_an_absent_run_blocks(self) -> None:
        """Absence is PENDING, which the caller's deadline converts to FAILURE."""
        payload = [
            row for row in _complete_external_payload() if row["name"] != GATE_CONTEXT
        ]
        code, report = evaluate(
            _all_gates_success(),
            check_runs=payload,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_PENDING, report
        assert GATE_CONTEXT in report

    def test_a_red_run_blocks(self) -> None:
        payload = [dict(row) for row in _complete_external_payload()]
        for row in payload:
            if row["name"] == GATE_CONTEXT:
                row["conclusion"] = "failure"
        code, report = evaluate(
            _all_gates_success(),
            check_runs=payload,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_FAILURE, report
        assert GATE_CONTEXT in report

    def test_the_tuple_entry_is_the_mechanism(self) -> None:
        """Falsification control: the same red payload passes once unregistered.

        Without this case the two above would also hold if some other surface
        happened to catch the failure, and the tuple entry could be deleted with
        every test still green.
        """
        payload = [dict(row) for row in _complete_external_payload()]
        for row in payload:
            if row["name"] == GATE_CONTEXT:
                row["conclusion"] = "failure"
        without = tuple(c for c in EXPECTED_EXTERNAL_CONTEXTS if c != GATE_CONTEXT)
        code, report = evaluate(
            _all_gates_success(), check_runs=payload, external_contexts=without
        )
        assert code == EXIT_SUCCESS, report
