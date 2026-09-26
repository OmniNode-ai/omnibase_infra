# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for queue-aware auto-merge arming (OMN-19697)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "auto-merge.yml"
)


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _enable_step() -> dict[str, Any]:
    steps = _load_workflow()["jobs"]["auto-merge"]["steps"]
    matches = [step for step in steps if step.get("name") == "Enable auto-merge"]
    assert len(matches) == 1
    return matches[0]


def test_enable_step_uses_queue_aware_arm_args() -> None:
    run = str(_enable_step()["run"])
    assert "merge_queue_enqueue.py arm-args" in run
    assert "mergeQueue(branch:" in run


def test_enable_step_does_not_hardcode_squash_auto_on_merge_command() -> None:
    merge_lines = [
        line
        for line in str(_enable_step()["run"]).splitlines()
        if "gh pr merge" in line
    ]
    assert merge_lines
    assert all("--squash --auto" not in line for line in merge_lines)


def test_enable_step_keeps_cross_repo_pat() -> None:
    assert _enable_step()["env"]["GH_TOKEN"] == "${{ secrets.CROSS_REPO_PAT }}"
