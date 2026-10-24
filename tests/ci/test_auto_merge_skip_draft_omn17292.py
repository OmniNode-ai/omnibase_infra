# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: auto-merge.yml does not try to arm a draft PR.

A draft's ``check_suite: completed`` event reached the Enable step, and GitHub
refused the arming call with "Pull request is a draft
(enablePullRequestAutoMerge)", which failed the Enable Auto-Merge job red on
dev (run 40034772634). The step now reads ``isDraft`` and returns before the
arming call, as the enqueue step's classifier already treats DRAFT as wait.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "auto-merge.yml"
_STEP_PREFIX = "auto-merge failed: "


def _enable_run() -> str:
    data: dict[Any, Any] = yaml.safe_load(_WORKFLOW.read_text())
    steps: list[dict[str, Any]] = data["jobs"]["auto-merge"]["steps"]
    found = [s for s in steps if s.get("name") == "Enable auto-merge"]
    assert len(found) == 1, "expected exactly one 'Enable auto-merge' step"
    return str(found[0]["run"])


def _run_enable(
    tmp_path: Path, *, draft: str, merge_output: str = "", merge_rc: int = 0
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the Enable step's script against a stub ``gh`` that logs its argv.

    ``gh pr merge`` prints ``merge_output`` and exits ``merge_rc``.
    """
    log = tmp_path / "gh.log"
    merge_out = tmp_path / "merge_output.txt"
    merge_out.write_text(merge_output)
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{log}"\n'
        'case "$*" in\n'
        "  *autoMergeRequest*) echo false ;;\n"
        f'  *isDraft*) echo "{draft}" ;;\n'
        "  *baseRefName*) echo dev ;;\n"
        '  *"api graphql"*) echo \'{"data":{"repository":{"mergeQueue":null}}}\' ;;\n'
        f'  *"pr merge"*) cat "{merge_out}"; exit {merge_rc} ;;\n'
        "esac\n"
    )
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "GH_REPO": "OmniNode-ai/omnibase_infra",
        "PR": "1",
    }
    result = subprocess.run(
        ["bash", "-c", _enable_run()],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = log.read_text() if log.exists() else ""
    return result, calls


def _recorded_refusal(recorded_response: dict[str, object]) -> str:
    response = cast("dict[str, object]", recorded_response["response"])
    step_output = cast("str", response["step_output"])
    assert step_output.startswith(_STEP_PREFIX)
    return step_output.removeprefix(_STEP_PREFIX)


def test_draft_read_precedes_the_arming_call() -> None:
    run = _enable_run()
    guard = run.find("--json isDraft")
    arm = run.find("gh pr merge")
    assert guard != -1, "Enable step must read isDraft"
    assert guard < arm, "the draft read must precede the arming call"
    assert "exit 0" in run[guard:arm]


@pytest.mark.live_contact("tests/ci/fixtures/auto_merge_draft_refusal_omn17292.json")
def test_recorded_draft_refusal_is_a_red_without_the_guard(
    recorded_response: dict[str, object], tmp_path: Path
) -> None:
    # GitHub's recorded answer for arming a draft is not one of the step's
    # benign outputs, so a draft that reaches gh pr merge fails the job.
    refusal = _recorded_refusal(recorded_response)
    result, _ = _run_enable(tmp_path, draft="false", merge_output=refusal, merge_rc=1)
    assert result.returncode == 1
    assert f"{_STEP_PREFIX}{refusal}" in result.stdout


@pytest.mark.live_contact("tests/ci/fixtures/auto_merge_draft_refusal_omn17292.json")
def test_draft_pr_exits_zero_without_arming(
    recorded_response: dict[str, object], tmp_path: Path
) -> None:
    refusal = _recorded_refusal(recorded_response)
    result, calls = _run_enable(
        tmp_path, draft="true", merge_output=refusal, merge_rc=1
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "is a draft; not arming" in result.stdout
    assert "pr merge" not in calls


def test_ready_pr_still_reaches_the_arming_call(tmp_path: Path) -> None:
    result, calls = _run_enable(tmp_path, draft="false")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pr merge" in calls
