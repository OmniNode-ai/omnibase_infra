# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shape-gate detector jobs run on every PR, whatever the preflight says (OMN-20298)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SUMMARY_GATE = ROOT / "scripts" / "ci" / "ci_summary_gate.py"

DETECTOR_JOBS = (
    "canonical-handler-shape-gate",
    "no-plugin-daemon-classes-gate",
    "shape-gate-independence",
)


def _jobs() -> dict[str, Any]:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    return jobs


@pytest.mark.unit
@pytest.mark.parametrize("job_id", DETECTOR_JOBS)
def test_detector_job_has_no_dependency_and_no_condition(job_id: str) -> None:
    job = _jobs()[job_id]
    assert "needs" not in job, f"{job_id} must not wait on any job, preflight included"
    assert "if" not in job, f"{job_id} must run on every event"


@pytest.mark.unit
@pytest.mark.parametrize("job_id", DETECTOR_JOBS)
def test_detector_job_is_a_strict_ci_summary_gate(job_id: str) -> None:
    display_name = _jobs()[job_id]["name"]
    assert f'"{display_name}"' in SUMMARY_GATE.read_text(encoding="utf-8")


@pytest.mark.unit
def test_lint_job_carries_no_shape_detector_step() -> None:
    steps = _jobs()["lint"]["steps"]
    text = " ".join(str(step) for step in steps)
    assert "canonical_handler_shape" not in text
    assert "no-plugin-daemon-classes" not in text
