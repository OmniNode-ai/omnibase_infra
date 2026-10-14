# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the whole-repo scans leave the commit path and stay enforced.

``check-direct-model-call`` (~40 s) and ``check-node-boundary-imports`` (~12 s)
each scan the whole repository, so they cost the same on a one-line commit as on
a large one. CI runs both on every pull request, so they run once per push here,
with growth judged against the merge base the way those CI jobs judge it (a
pre-push tree equals HEAD, so ``--base HEAD`` would judge nothing).

Falsifier: restore ``--base HEAD`` or the pre-commit stage on either hook, or
drop the CI job that backs it, and the named test fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
MERGE_BASE_BASE = '--base "$(git merge-base HEAD origin/dev)"'


def _hook(hook_id: str) -> dict[str, Any]:
    config = yaml.safe_load(PRECOMMIT.read_text(encoding="utf-8"))
    hooks = [
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("id") == hook_id
    ]
    assert len(hooks) == 1, f"exactly one {hook_id} hook is declared"
    return hooks[0]


@pytest.mark.parametrize(
    "hook_id", ["check-direct-model-call", "check-node-boundary-imports"]
)
def test_whole_repo_scan_runs_at_push_against_the_merge_base(hook_id: str) -> None:
    hook = _hook(hook_id)
    assert hook["stages"] == ["pre-push"]
    assert MERGE_BASE_BASE in hook["entry"]
    assert "--base" not in hook.get("args", []), (
        "the merge-base --base lives in the entry; a second --base would win or lose"
    )


def test_direct_model_call_ci_backstop_runs_on_every_pull_request() -> None:
    workflow = yaml.safe_load(
        (WORKFLOWS / "direct-model-call.yml").read_text(encoding="utf-8")
    )
    triggers = workflow.get("on", workflow.get(True))
    assert "pull_request" in triggers
    job = workflow["jobs"]["direct-model-call"]
    assert "if" not in job


def test_node_boundary_ci_backstop_runs_unconditionally() -> None:
    workflow = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["node-boundary-imports"]
    assert "if" not in job
    assert "needs" not in job
