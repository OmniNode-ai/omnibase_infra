# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``make up-local`` reaches a healthy runtime (OMN-22291 AC5).

The incident this closes was not a test failure. On 2026-11-03 the local stack
came up with every component healthy and ``make up-local`` still failed, three
times over 105 minutes, on ``dependency failed to start: container
omnibase-infra-local-omninode-runtime is unhealthy`` -- because the runtime's
own ``/health`` body reported ``status: unhealthy`` through one CRITICAL
dimension, ``consumer_sync``, on three quiet consumer groups.

So the receipt this AC asks for is the runtime's own status. The test reads the
CI job that actually boots the laptop profile on a runner: it asserts the job
boots with ``make up-local`` and then waits on ``.status == "healthy"`` from
each kernel's own ``/health`` body, which is precisely the field the incident
drove to ``unhealthy``. That job's run on this change is the standing live
proof, and it needs no local Docker. This file holds no live-stack test: one
that skips wherever there is no stack is collected and never executed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "reusable-runtime-boot.yml"
_JOB = "boot-catalog-local"


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return cast("list[dict[str, Any]]", workflow["jobs"][_JOB]["steps"])


def test_the_boot_job_boots_with_up_local_and_waits_on_the_runtime_health() -> None:
    """The job that boots the laptop profile gates on the runtime's own status."""
    runs = [step.get("run", "") for step in _steps()]
    assert any("make up-local" in run for run in runs), (
        f"{_JOB} is the AC5 falsifier only while it boots the profile the "
        "participant boots"
    )
    gate = [run for run in runs if "/health" in run and "status" in run]
    assert gate, f"{_JOB} must read the runtime's own /health body"
    body = "\n".join(gate)
    assert 'status" = "healthy"' in body or "status\" = 'healthy'" in body, (
        "the gate must require the runtime's own status to be healthy -- the "
        "incident reached unhealthy through runtime_health alone, with every "
        "component green"
    )
    assert "omninode-runtime" in body and "runtime-effects" in body, (
        "both kernels are gated: the incident's sibling effects runtime was "
        "in sync on the same stack, which is what proved the grading differed "
        "rather than the stack"
    )
    assert "::error::" in body, "a timeout must fail the job, not pass it quietly"
