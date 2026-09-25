# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17492: the hostile reviewer runs where its models are, and a review
that established no verdict is a named failure, never a success.

Measured 2026-09-25 over the 32 omnibase_infra pull requests merged between
2026-09-24T23:18Z and 11:10Z: the "Hostile Reviewer (adversarial gate)" job
concluded ``success`` on 28 heads and ``skipped`` on 4, and every one of the
28 success summaries read ``DEGRADED`` with ``Models succeeded: glm-review``
only. Run 36111018598 (omnibase_infra#4110) shows why, in order:

1. The job's trusted branch read ``OMNI_TRUSTED_CI_RUNS_ON_JSON``, which is
   ``["ubuntu-latest"]`` at org and repo scope, so the job ran on a hosted
   runner ("GitHub Actions 1000815865") that cannot open a TCP connection to
   the lab endpoint both local reviewer keys resolve to (3 of 3 attempts
   failed for each key).
2. glm-review alone succeeded, with seven findings including one error.
3. The quorum (OMN-18479) needs two succeeding models, so the reviewer
   returned ``degraded_quorum``, posted zero threads (``below_quorum: 7``),
   and the step exited 0, so the job and the "Hostile Review Gate" were
   green on a review that established nothing.

omniclaude fixed the same placement defect in OMN-18415 with a dedicated
variable whose fallback is the LAN-attached fleet; omnimarket fails the job
on any verdict other than ``passed`` (OMN-15110). This file pins both here.

The properties are asserted over the parsed workflow, and the verdict step is
executed as the shell it is, so a rewrite that keeps the text but loses the
behaviour still fails.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "hostile-reviewer.yml"
REVIEW_JOB = "hostile-review"
ENFORCE_STEP = "Enforce hostile-review verdict"
DEDICATED_VARIABLE = "vars.OMNI_HOSTILE_REVIEW_RUNS_ON_JSON"
LAN_FLEET_FALLBACK = """'["self-hosted","omnibase-ci"]'"""

pytestmark = pytest.mark.unit


def _review_job() -> dict[str, Any]:
    with WORKFLOW.open(encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    job = parsed["jobs"].get(REVIEW_JOB)
    assert isinstance(job, dict), f"{REVIEW_JOB} is absent from {WORKFLOW.name}"
    return job


def _enforce_step() -> dict[str, Any]:
    steps = _review_job()["steps"]
    matches = [s for s in steps if str(s.get("name", "")).startswith(ENFORCE_STEP)]
    assert len(matches) == 1, (
        f"expected exactly one '{ENFORCE_STEP}' step in {REVIEW_JOB}, found "
        f"{len(matches)}"
    )
    return matches[0]


def test_trusted_branch_reads_the_dedicated_variable_with_the_lan_fallback() -> None:
    runs_on = str(_review_job()["runs-on"])
    assert DEDICATED_VARIABLE in runs_on
    assert f"{DEDICATED_VARIABLE} || {LAN_FLEET_FALLBACK}" in runs_on, (
        "the trusted branch must fall back to the LAN-attached fleet, where "
        "the local reviewer keys are reachable"
    )


def test_trusted_branch_no_longer_reads_the_hosted_seam() -> None:
    runs_on = str(_review_job()["runs-on"])
    assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" not in runs_on, (
        "the shared trusted seam is ubuntu-latest, where no local reviewer "
        "key is reachable; that placement is the defect this test pins"
    )


def test_fork_pull_requests_still_take_the_public_path_first() -> None:
    runs_on = str(_review_job()["runs-on"])
    fork_test = "github.event.pull_request.head.repo.full_name != github.repository"
    assert fork_test in runs_on
    assert runs_on.index("OMNI_PUBLIC_PR_RUNS_ON_JSON") < runs_on.index(
        DEDICATED_VARIABLE
    ), "fork code must never be routed to the LAN-attached fleet"


def test_enforce_step_runs_even_when_an_earlier_step_failed() -> None:
    condition = str(_enforce_step().get("if", ""))
    assert "always()" in condition or "!cancelled()" in condition


def test_enforce_step_is_the_last_step_of_the_job() -> None:
    steps = _review_job()["steps"]
    assert str(steps[-1].get("name", "")).startswith(ENFORCE_STEP), (
        "the verdict must be enforced after the summary comment posts, so a "
        "degraded run still explains itself on the pull request"
    )


def test_enforce_step_reads_both_verdict_producers() -> None:
    env = _enforce_step().get("env", {})
    source = str(env.get("REVIEW_VERDICT", ""))
    assert "steps.review.outputs.verdict" in source
    assert "steps.preflight.outputs.verdict" in source


def _run_enforce(
    verdict: str | None, error: str = ""
) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin", "REVIEW_ERROR": error}
    if verdict is not None:
        env["REVIEW_VERDICT"] = verdict
    return subprocess.run(
        ["bash", "-c", str(_enforce_step()["run"])],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_passed_verdict_succeeds() -> None:
    result = _run_enforce("passed")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("verdict", ["degraded", "", None, "blocked", "unknown"])
def test_every_other_verdict_is_a_named_failure(verdict: str | None) -> None:
    result = _run_enforce(verdict, error="all review endpoints unreachable")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error::" in result.stdout
    assert "no review verdict" in result.stdout.lower()


def test_the_failure_names_the_upstream_error() -> None:
    result = _run_enforce("degraded", error="cli_review exit 2")
    assert "cli_review exit 2" in result.stdout
