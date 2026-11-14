# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy-agent test job runs in parallel inside its unchanged limit (OMN-24428).

The deploy-agent suite is 2300+ tests and took about 8.5 minutes serial against
the job's 10 minute limit. On a loaded runner the job was cancelled at its
limit and CI Summary failed on pull requests whose own changes were fine. The
fix is not a longer limit: the job runs on four parallel workers grouped by file, and
its ``timeout-minutes`` stays what it was.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-agent-tests.yml"
AGENT_PYPROJECT = REPO_ROOT / "scripts" / "deploy-agent" / "pyproject.toml"
JOB_TIMEOUT_MINUTES = 10
SUITE_STEP = "Run deploy-agent unit tests (OMN-15378)"


def _suite_job() -> dict[str, Any]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job: dict[str, Any] = workflow["jobs"]["deploy-agent-tests"]
    return job


@pytest.mark.unit
def test_suite_step_runs_on_parallel_workers_grouped_by_file() -> None:
    steps = {step.get("name"): step for step in _suite_job()["steps"]}
    run = steps[SUITE_STEP]["run"]
    assert "-n 4" in run, "the deploy-agent suite must run on four parallel workers"
    assert "-n auto" not in run, "a worker per core OOM-kills a many-core host"
    assert "--dist loadfile" in run, "tests of one file must share a worker"


@pytest.mark.unit
def test_job_limit_is_not_widened() -> None:
    assert _suite_job()["timeout-minutes"] == JOB_TIMEOUT_MINUTES


@pytest.mark.unit
def test_dev_extra_declares_the_parallel_runner() -> None:
    with AGENT_PYPROJECT.open("rb") as handle:
        dev = tomllib.load(handle)["project"]["optional-dependencies"]["dev"]
    assert any(dep.startswith("pytest-xdist") for dep in dev)
