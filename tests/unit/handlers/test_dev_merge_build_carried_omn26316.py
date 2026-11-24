# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev job counts as building a merge only when it carried it (OMN-26316).

The records are the agent's job records of omnibase_infra#6191 (merged 17:46:52Z on
2026-11-23). b32e8022 is the stability-lane follow job accepted at 18:00:21Z: its image
(425e6ea5) never put 6191 on dev, and its first real dev build is job 2098b18c (completed
19:13:49Z). The field that separates them is the job's built sha, the lineage record's
resolved ref, judged by ancestry against the merge: a job whose built sha is unread is not
counted (fail closed), and a stability-follow job is never counted. Each test names its clock.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_dev_merge_build_overdue import (
    HandlerDevMergeBuildOverdue,
)
from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)
from omnibase_infra.models.model_overdue_dev_merge_build_request import (
    ModelOverdueDevMergeBuildRequest,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE = (
    _ROOT
    / "tests/fixtures/overdue_dev_merge_build_omn26316/merge_6191_job_records.json"
)
_STABILITY_FOLLOW = "agent/stability-follow-tag"
_PRE = "1" * 40
_POST = "d" * 40


def _decide(
    now: str,
    *,
    drop: tuple[str, ...] = (),
    replace: dict[str, dict[str, Any]] | None = None,
) -> ModelOverdueDevMergeBuildDecision:
    record = json.loads(_FIXTURE.read_text())
    merge = record["merge"]
    jobs = []
    for job in record["jobs"]:
        key = job["job_id"][:8]
        if key in drop:
            continue
        jobs.append(
            {
                **{k: v for k, v in job.items() if k != "completed_at"},
                **(replace or {}).get(key, {}),
            }
        )
    return HandlerDevMergeBuildOverdue().handle(
        ModelOverdueDevMergeBuildRequest.model_validate(
            {
                "repository": merge["repository"],
                "merge_sha": merge["merge_sha"],
                "merged_at": merge["merged_at"],
                "runtime_class": merge["runtime_class"],
                "now": now,
                "window_seconds": 3600,
                "jobs": jobs,
                "descendant_shas": merge["descendant_shas"],
            }
        )
    )


def test_the_fixture_holds_the_stability_job_beside_the_dev_jobs() -> None:
    ids = [job["job_id"][:8] for job in json.loads(_FIXTURE.read_text())["jobs"]]
    assert "b32e8022" in ids


def test_the_stability_follow_job_never_reads_6191_built_at_18_13() -> None:
    """Worst case: the follow job's record names the dev lane, the dev head and success."""
    decision = _decide(
        "2026-11-23T18:13:00Z",
        drop=("2098b18c",),
        replace={"b32e8022": {"runtime_lane": "dev", "built_sha": _POST}},
    )
    assert decision.outcome == "WAIT", decision
    assert decision.reason == "within_window"


def test_the_stability_follow_job_does_not_stop_the_request_at_18_50() -> None:
    decision = _decide(
        "2026-11-23T18:50:00Z",
        drop=("2098b18c",),
        replace={"b32e8022": {"runtime_lane": "dev", "built_sha": _POST}},
    )
    assert (decision.outcome, decision.reason) == ("REQUEST", "overdue_no_build")


def test_a_dev_job_whose_built_sha_is_unread_is_not_counted() -> None:
    decision = _decide("2026-11-23T19:14:00Z", replace={"2098b18c": {"built_sha": ""}})
    assert decision.outcome == "REQUEST"


def test_a_dev_job_accepted_after_the_merge_that_built_an_older_head_is_not_counted() -> (
    None
):
    """Accepted after the merge, git_ref origin/dev, success, yet its sha lacks the merge."""
    decision = _decide(
        "2026-11-23T19:14:00Z", replace={"2098b18c": {"built_sha": _PRE}}
    )
    assert decision.outcome == "REQUEST"


def test_the_first_real_dev_build_2098b18c_reads_built() -> None:
    decision = _decide("2026-11-23T19:14:00Z")
    assert (decision.outcome, decision.reason) == ("SATISFIED", "built_by_job")
    assert str(decision.job_id).startswith("2098b18c")


def test_a_stability_follow_requester_is_never_counted_even_with_a_carrying_sha() -> (
    None
):
    decision = _decide(
        "2026-11-23T19:14:00Z",
        replace={"2098b18c": {"requested_by": _STABILITY_FOLLOW}},
    )
    assert decision.outcome == "REQUEST"


def test_the_caller_proves_ancestry_with_git(tmp_path: Path) -> None:
    from omnibase_infra.nodes.node_deploy_job_reading_effect import (
        runtime_deploy_job_reading as caller,
    )

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    git("commit", "--allow-empty", "-q", "-m", "older")
    older = git("rev-parse", "HEAD")
    git("commit", "--allow-empty", "-q", "-m", "merge")
    merge = git("rev-parse", "HEAD")
    git("commit", "--allow-empty", "-q", "-m", "newer")
    newer = git("rev-parse", "HEAD")
    unknown = "e" * 40
    carried = caller.built_shas_carrying(
        str(tmp_path), merge, {older, merge, newer, unknown, ""}, 10.0
    )
    assert carried == {merge, newer}
