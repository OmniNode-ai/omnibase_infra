# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22275: CI Summary's external sweep reads an accepted Hostile Review run as success.

``Hostile Reviewer (adversarial gate)`` and ``Hostile Review Gate`` are named by nothing
else in CI Summary, so the sweep (``evaluate_external_sweep``) is what turns a red one into
a red umbrella. The workflow's own steps are run (see ``_hostile_review_job_omn22275``) for
each voter result, and the conclusion of each context is written as the check-run row the
sweep reads. An accepted agent receipt leaves both rows ``success``; the sweep then
files no failure. A degraded run with no accepted receipt, and a degraded run whose answering voter raised a blocking finding, leave a row
``failure`` and the sweep names it: the sweep itself is unchanged and still fails closed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.ci_summary_gate import (
    EXPECTED_EXTERNAL_CONTEXTS,
    EXTERNAL_SWEEP_EXCLUSIONS,
    evaluate_external_sweep,
)
from tests.ci._hostile_review_job_omn22275 import (
    HEAD_SHA,
    OTHER_HEAD_SHA,
    Run,
    reported_contexts,
    simulate,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 3, 6, 0, 0, tzinfo=UTC)
ROOT, GATE = reported_contexts()


def _rows(run: Run) -> list[dict[str, Any]]:
    return [
        {
            "id": 1000 + i,
            "name": name,
            "status": "completed",
            "conclusion": "success" if green else "failure",
            "started_at": "2026-11-03T05:00:00Z",
            "completed_at": "2026-11-03T05:10:00Z",
            "head_sha": HEAD_SHA,
        }
        for i, (name, green) in enumerate(sorted(run.contexts.items()))
    ]


def _sweep(run: Run, *, exclusions: dict[str, Any]) -> tuple[list[str], list[str]]:
    failures, _in_flight, swept, _excluded, _provisional = evaluate_external_sweep(
        _rows(run),
        expected=EXPECTED_EXTERNAL_CONTEXTS,
        in_run_names=frozenset(),
        self_name="CI Summary",
        exclusions=exclusions,
        events={},
        now=NOW,
    )
    return failures, swept


def _run(tmp_path: Path, case: str, **kwargs: Any) -> Run:
    work = tmp_path / case
    work.mkdir()
    return simulate(work, **kwargs)


@pytest.mark.parametrize(
    "registry", [{}, EXTERNAL_SWEEP_EXCLUSIONS], ids=["bare", "live"]
)
def test_ci_summary_external_sweep_reads_the_accepted_run_as_success(
    tmp_path: Path, registry: dict[str, Any]
) -> None:
    run = _run(tmp_path, "accepted", verdict="degraded", receipt_head=HEAD_SHA)
    assert run.outputs["accept"]["decision"] == "pass_agent"
    failures, swept = _sweep(run, exclusions=registry)
    assert failures == [], failures
    if not registry:
        assert GATE in swept, (
            "the sweep really looked at the gate (zero needs a control)"
        )


def test_ci_summary_external_sweep_still_names_a_degraded_run_with_no_receipt(
    tmp_path: Path,
) -> None:
    """POSITIVE CONTROL: the sweep that passes the accepted run fails the unaccepted one."""
    run = _run(tmp_path, "norcpt", verdict="degraded")
    failures, _swept = _sweep(run, exclusions={})
    assert f"{GATE} (failure)" in failures, failures
    assert f"{ROOT} (failure)" in failures, failures


def test_ci_summary_external_sweep_still_names_a_receipt_for_another_head(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "otherhead", verdict="degraded", receipt_head=OTHER_HEAD_SHA)
    failures, _swept = _sweep(run, exclusions={})
    assert f"{GATE} (failure)" in failures, failures


def test_ci_summary_external_sweep_still_names_a_blocking_finding_with_a_receipt(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "blocking", verdict="degraded_blocking", receipt_head=HEAD_SHA)
    failures, _swept = _sweep(run, exclusions={})
    assert f"{GATE} (failure)" in failures, failures
