# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``make up-local`` reaches a healthy runtime (OMN-22291 AC5).

The incident this closes was not a test failure. On 2026-11-03 the local stack
came up with every component healthy and ``make up-local`` still failed, three
times over 105 minutes, on ``dependency failed to start: container
omnibase-infra-local-omninode-runtime is unhealthy`` -- because the runtime's
own ``/health`` body reported ``status: unhealthy`` through one CRITICAL
dimension, ``consumer_sync``, on three quiet consumer groups.

So the receipt this AC asks for is the runtime's own status, and there are two
places it can be read. The first test reads the CI job that actually boots the
laptop profile on a runner: it asserts the job boots with ``make up-local`` and
then waits on ``.status == "healthy"`` from each kernel's own ``/health`` body,
which is precisely the field the incident drove to ``unhealthy``. That job's
run on this change is the standing live proof, and it needs no local Docker.

The second test reads a live stack when one is on the machine running the
tests, and is skipped -- never passed -- when there is none. A skip is not a
pass: on a machine with no stack this AC is proven by the CI job above and by
nothing in this file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "reusable-runtime-boot.yml"
_JOB = "boot-catalog-local"
_PROJECT = os.environ.get("LOCAL_PROJECT", "omnibase-infra-local")
_RUNTIME_CONTAINER = f"{_PROJECT}-omninode-runtime"
_HEALTH_URL = "http://localhost:8085/health"


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


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _live_stack_or_skip() -> None:
    if shutil.which("docker") is None:
        pytest.skip(
            "no docker on this machine, so there is no local stack to read; "
            "OMN-22291 AC5 is proven live by the boot-catalog-local job"
        )
    probe = _docker("inspect", "--format", "{{.State.Status}}", _RUNTIME_CONTAINER)
    if probe.returncode != 0:
        pytest.skip(
            f"{_RUNTIME_CONTAINER} is not on this machine "
            f"({probe.stderr.strip()[:200]}), so there is no stack to read; "
            "OMN-22291 AC5 is proven live by the boot-catalog-local job"
        )


def test_up_local_reaches_healthy() -> None:
    """A live local stack's runtime reports healthy, and says so itself."""
    _live_stack_or_skip()

    health = _docker(
        "exec",
        _RUNTIME_CONTAINER,
        "curl",
        "-sS",
        "--max-time",
        "10",
        _HEALTH_URL,
    )
    assert health.returncode == 0, (
        f"the runtime's own health endpoint did not answer: "
        f"{health.stderr.strip()[:400]}"
    )
    payload = json.loads(health.stdout)

    runtime_health = payload.get("runtime_health") or {}
    dimensions = runtime_health.get("dimensions") or {}
    consumer_sync = dimensions.get("consumer_sync") or {}
    assert consumer_sync.get("status") != "CRITICAL", (
        "consumer_sync graded CRITICAL on this stack, which is the incident: "
        f"{consumer_sync.get('detail', '<no detail>')}"
    )
    assert payload.get("status") == "healthy", (
        "the receipt is the runtime's own status, and compose makes every "
        f"dependent wait on it: status={payload.get('status')!r} "
        f"runtime_health={runtime_health.get('status')!r} "
        f"consumer_sync={consumer_sync.get('detail', '<no detail>')!r}"
    )
