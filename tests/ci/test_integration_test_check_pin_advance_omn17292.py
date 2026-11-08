# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: Integration Test Coverage exempts the omnimarket pin-advance bot.

The bot regenerates ``src/omnibase_infra/topology/table_grant_derivation.py``
from the upstream pin. That is derived code, not a feature, so the gate must
not demand an integration test for it. Other branches, and forks, still reach
the git-diff logic.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "integration-test-check.yml"
)
_EXEMPT_MESSAGE = "Omnimarket contract-pin advance PR"


def _run_step(
    head_ref: str, head_repo: str, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["check-integration-tests"]["steps"]
    step = next(
        s for s in steps if s.get("name") == "Check for integration tests in PR"
    )
    env = {
        "PATH": os.environ["PATH"],
        "GITHUB_REPOSITORY": "OmniNode-ai/omnibase_infra",
        "GITHUB_BASE_REF": "dev",
        "PR_HEAD_REF": head_ref,
        "PR_HEAD_REPO": head_repo,
    }
    return subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pin_advance_bot_branch_is_exempt(tmp_path: Path) -> None:
    result = _run_step(
        "bot/omnimarket-contract-pin-20261108063222",
        "OmniNode-ai/omnibase_infra",
        tmp_path,
    )
    assert result.returncode == 0
    assert _EXEMPT_MESSAGE in result.stdout


@pytest.mark.parametrize(
    ("head_ref", "head_repo"),
    [
        ("omn-22712-venv-additive-atomic", "OmniNode-ai/omnibase_infra"),
        ("bot/omnimarket-contract-pin-20261108063222", "someone/omnibase_infra"),
    ],
)
def test_other_branches_and_forks_are_not_exempt(
    head_ref: str, head_repo: str, tmp_path: Path
) -> None:
    result = _run_step(head_ref, head_repo, tmp_path)
    assert _EXEMPT_MESSAGE not in result.stdout
