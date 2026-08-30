# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17199: the exposure-reader gate must be able to block a merge.

CLAUDE.md Operating Rule 5 — a check that is not a merge condition is advisory
and gets ignored. The ticket says it directly: *a validator without its gate has
failed this ticket*. So the wiring is asserted here, not assumed.

On this repo the wiring is a two-part claim and the first part is a trap. Live
readback on 2026-08-30::

    gh api repos/OmniNode-ai/omnibase_infra/branches/dev/protection/required_status_checks \\
        --jq '.contexts'
    ["CI Summary"]

`dev` requires exactly ONE context. Every other gate in this repo — including
``dispatcher-route-coverage``, whose row in ``.github/required-checks.yaml``
still claims to be a "live required status check on omnibase_infra dev branch
protection" — is enforced through :data:`EXPECTED_EXTERNAL_CONTEXTS`, the tuple
``CI Summary`` asserts present-completed-success on the PR head. That tuple IS
the enforcement surface (OMN-4497 single-umbrella design, restated by OMN-16878),
and CLAUDE.md's "PR CI Requirements" §4 records the same thing.

So membership in that tuple is what makes this gate blocking, and these tests
pin it along with the producer-side properties that make membership meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXIT_PENDING,
    EXIT_SUCCESS,
    EXPECTED_EXTERNAL_CONTEXTS,
    EXTERNAL_GOOD_CONCLUSIONS,
    MEASURED_NOT_ENFORCED_CONTEXTS,
    POST_FIXTURE_WINDOW_CONTEXTS,
    evaluate,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

GATE_CONTEXT = "exposure-reader-coverage"
GATE_WORKFLOW = "exposure-reader-coverage.yml"


def _workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict[str, Any]) -> set[str]:
    # PyYAML parses a bare `on:` key as the boolean True.
    raw = workflow.get(True, workflow.get("on"))
    if isinstance(raw, dict):
        return set(raw)
    if isinstance(raw, list):
        return set(raw)
    return {str(raw)}


def _all_gates_success() -> list[dict[str, Any]]:
    """A full, passing in-run snapshot: every strict AND skippable gate green.

    Both tuples, not just the strict one: ``evaluate`` requires a skippable gate to be
    present-and-good or explicitly skipped, so a strict-only payload reports five absent
    gates and the assertion under test would be measuring the wrong thing.
    """
    from scripts.ci.ci_summary_gate import SKIPPABLE_GATE_JOBS, STRICT_GATE_JOBS

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
        """AC3 — membership in the tuple is the ONLY thing making this block.

        Dropping it does not downgrade the gate to a weaker surface; it removes
        the gate entirely, with nothing in branch protection to reveal the loss.
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

    def test_the_producer_has_no_skip_path(self) -> None:
        """No `needs:`, no job-level `if:`, no path filter.

        A skipped producer is the OMN-15057 vector-5 silent pass. It is defused
        twice over (EXTERNAL_GOOD_CONCLUSIONS admits only "success"), but the
        cheapest place to keep it defused is the producer.
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


class TestPostFixtureWindowContexts:
    """The historical-fixture exclusion must not become a runtime bypass.

    ``POST_FIXTURE_WINDOW_CONTEXTS`` exists so historical replays are not fed
    fabricated check-run rows for merged PRs that never ran a workflow which did
    not yet exist. If it ever started weakening the live assertion it would be
    the very thing this ticket was filed about: an escape hatch that can be
    taken silently.
    """

    def test_every_excluded_context_is_actually_asserted(self) -> None:
        assert set(EXPECTED_EXTERNAL_CONTEXTS) >= POST_FIXTURE_WINDOW_CONTEXTS

    def test_the_exclusion_is_not_the_whole_tuple(self) -> None:
        assert set(EXPECTED_EXTERNAL_CONTEXTS) != POST_FIXTURE_WINDOW_CONTEXTS

    def test_a_complete_payload_greens(self) -> None:
        """Guard the premise of the next test: the payload is otherwise clean."""
        code, report = evaluate(
            _all_gates_success(),
            check_runs=_complete_external_payload(),
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_SUCCESS, report

    @pytest.mark.parametrize("context", sorted(POST_FIXTURE_WINDOW_CONTEXTS))
    def test_an_excluded_context_still_blocks_when_absent(self, context: str) -> None:
        """Absence must be PENDING (→ FAILURE on deadline), never SUCCESS."""
        payload = [
            row for row in _complete_external_payload() if row["name"] != context
        ]
        code, report = evaluate(
            _all_gates_success(),
            check_runs=payload,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_PENDING, report
        assert context in report

    @pytest.mark.parametrize("context", sorted(POST_FIXTURE_WINDOW_CONTEXTS))
    def test_an_excluded_context_still_blocks_when_red(self, context: str) -> None:
        from scripts.ci.ci_summary_gate import EXIT_FAILURE

        payload = [dict(row) for row in _complete_external_payload()]
        for row in payload:
            if row["name"] == context:
                row["conclusion"] = "failure"
        code, report = evaluate(
            _all_gates_success(),
            check_runs=payload,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_FAILURE, report
        assert context in report


class TestTheJobCanActuallyImportTheValidator:
    """A gate that is structurally red forever is a gate people learn to ignore.

    The validator is a module under ``src/omnibase_infra/validators/`` rather than a
    ``scripts/**`` file (OMN-14475 default-deny), so importing it executes
    ``omnibase_infra/__init__.py``, which imports ``omnibase_core.enums``. The first
    cut of this workflow installed only ``pyyaml`` and died on every run with
    ``ModuleNotFoundError: No module named 'omnibase_core'`` (PR #3048, run
    33335363199) — failing closed, which is correct, but for an infrastructure reason
    rather than a real violation. A permanently-red required gate is the OMN-14440
    shape from the other direction: output nobody can act on.
    """

    WORKFLOW = "exposure-reader-coverage.yml"

    def _steps(self) -> list[dict[str, Any]]:
        workflow = _workflow(self.WORKFLOW)
        return list(workflow["jobs"]["exposure-reader-coverage"]["steps"])

    def test_the_job_installs_the_repo_dependency_environment(self) -> None:
        uses = [str(step.get("uses", "")) for step in self._steps()]
        assert any("setup-python-uv" in entry for entry in uses), (
            "the job must set up the repo's own uv environment; the validator's "
            "import chain reaches omnibase_core and a bare `pip install pyyaml` "
            "makes this gate red on every run"
        )

    def test_the_gate_runs_the_validator_through_that_environment(self) -> None:
        runs = "\n".join(str(step.get("run", "")) for step in self._steps())
        assert (
            "uv run python -m omnibase_infra.validators.bus_backed_exposure_readers"
            in runs
        ), "the validator must be invoked through the resolved uv environment"

    def test_the_dependency_pins_are_not_restated_in_the_workflow(self) -> None:
        """No second source of truth for the core/spi/compat pins.

        Re-pinning them inline here is the exact drift shape this gate exists to
        catch: two declarations of one fact, free to diverge silently.
        """
        text = (WORKFLOWS / self.WORKFLOW).read_text(encoding="utf-8")
        for package in ("omnibase-core==", "omnibase-spi==", "omnibase-compat=="):
            assert package not in text, (
                f"{package} is pinned in pyproject.toml/uv.lock; restating it in "
                "the workflow creates a second source of truth"
            )
