# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The own-repository lag bound runs through the real deploy job reading node (OMN-26316).

The agent's own test suite runs in its own project environment, where ``deploy_agent`` is
installed and the agent loads ``runtime_deploy_job_reading`` by path exactly as it does on
dev. Only the ledger append command and the state directory are replaced by the suite's
fixtures; the node contract, the routing table and the host composition root are the
checkout's own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_TESTS = "scripts/deploy-agent/tests/unit/test_own_repo_lag_bound_omn26316.py"


def test_own_repo_lag_bound_runs_through_the_agents_own_loader() -> None:
    done = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--project",
            "scripts/deploy-agent",
            "--extra",
            "dev",
            "pytest",
            AGENT_TESTS,
            "-q",
            "-p",
            "no:randomly",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
