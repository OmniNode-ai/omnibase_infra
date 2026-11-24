# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20413: Enable Auto-Merge does not fail for a PR that edits a workflow.

The arming token is a PAT without ``workflow`` scope (OMN-17875), so GitHub
refuses ``enablePullRequestAutoMerge`` on any PR that touches
``.github/workflows/*`` and the job went red, which failed CI Summary too
(omnibase_infra#6218). The refusal is a property of the PR's files, so the step
warns, leaves the PR unarmed and skips the enqueue step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.ci.test_auto_merge_skip_draft_omn17292 import _run_enable

pytestmark = pytest.mark.unit

_WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "auto-merge.yml"
)
_REFUSAL = (
    "GraphQL: Pull request refusing to allow a Personal Access Token to create or "
    "update workflow `.github/workflows/chain-canary.yml` without `workflow` scope "
    "(enablePullRequestAutoMerge)"
)


def test_workflow_scope_refusal_is_not_a_failure(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    out.touch()
    result, _ = _run_enable(
        tmp_path,
        draft="false",
        merge_output=_REFUSAL,
        merge_rc=1,
        extra_env={"GITHUB_OUTPUT": str(out)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow_scope_refused=true" in out.read_text()


def test_other_refusals_still_fail(tmp_path: Path) -> None:
    result, _ = _run_enable(
        tmp_path, draft="false", merge_output="GraphQL: something else", merge_rc=1
    )
    assert result.returncode == 1


def test_enqueue_step_is_skipped_after_the_refusal() -> None:
    data: dict[Any, Any] = yaml.safe_load(_WORKFLOW.read_text())
    steps = {s["name"]: s for s in data["jobs"]["auto-merge"]["steps"] if "name" in s}
    cond = steps["Enqueue armed PR and verify it entered the queue"]["if"]
    assert "steps.arm.outputs.workflow_scope_refused != 'true'" in cond
    assert steps["Enable auto-merge"]["id"] == "arm"
