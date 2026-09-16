# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the VERIFY/CRON jobs moved off the single deploy runner (OMN-18408).

`omnibase-deploy` has exactly one member, `omninode-deploy-runner`, because
that container alone holds the private OMNI_HOME clone tree the release-train
deploy scripts write to (OMN-14889/OMN-14900). Every job pinned to that label
therefore serialises behind whatever else is running on it -- measured
2026-09-15T20:00Z as 12 queued plus 1 in-progress run against one busy runner,
while the 88-runner `omnibase-ci` fleet sat idle.

Five scheduled/per-merge VERIFY probes were on that label only because they
need the lab host's docker socket and its `host.docker.internal` gateway, not
because they write anything. They moved to `omnibase-verify`. The DEPLOY jobs
did not, and neither did four other verify jobs the operator deliberately left
serialised.

The scope here is exact, in both directions. The moved set is pinned so a
later edit cannot silently widen it, and the LEFT-BEHIND set is pinned so a
later edit cannot silently drain the deploy label of the jobs that must stay
serialised against a release-train deploy (CLAUDE.md rule 2a/12).

Each assertion resolves the job's own `runs-on` out of parsed YAML rather than
grepping the file: a grep cannot tell a live `runs-on` from one inside a
comment, and every one of these files carries comments that name both labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Both labels, always. `omnibase-verify` is a runner CLASS; `host-201` is the
# HOST. A second verify-class runner (omninode-mini-runner-1, arch-arm64) came
# online on another lab host on 2026-09-16 and is a legal match for the class
# alone -- and it cannot see the .201 lane these five jobs probe, so a run
# placed there reports the lane unreachable rather than failing to schedule.
VERIFY_LABEL = ["self-hosted", "omnibase-verify", "host-201"]
HOST_LABEL = "host-201"
DEPLOY_LABEL = ["self-hosted", "omnibase-deploy"]

# (workflow file, job key, job `name:`) -- the five named in the OMN-18408
# acceptance criteria and in the authorising operator-consent row.
MOVED_JOBS = (
    ("dev-lane-liveness.yml", "dev-lane-liveness", "dev-lane-liveness"),
    ("dev-lane-staleness.yml", "dev-lane-staleness", "dev-lane-staleness"),
    ("chain-canary.yml", "chain-canary", "Chain Canary (dev lane)"),
    ("msk-bastion-canary.yml", "bastion-canary", "MSK bastion routing canary"),
    (
        "runtime-rebuild-trigger.yml",
        "verify-lab-overlay-converged",
        "Verify the k3s onex-lab overlay applied the merged sha",
    ),
)

# Deliberately NOT moved. The first is the release-train DEPLOY job that must
# stay serialised; the rest are verify jobs the operator left on the deploy
# label pending a separate decision. A change that moves one of these is a
# scope change and must fail here rather than pass quietly.
STAYED_JOBS = (
    ("release-train-lab.yml", "deploy"),
    ("release-train-lab.yml", "cut-tag"),
    ("runtime-rebuild-trigger.yml", "verify-lane-converged"),
    ("runtime-rebuild-trigger-reusable.yml", "verify-sibling-converged"),
)


def _jobs(workflow: str) -> dict[str, Any]:
    path = WORKFLOWS / workflow
    assert path.is_file(), f"{path} does not exist"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{workflow} did not parse as a mapping"
    jobs = loaded.get("jobs")
    assert isinstance(jobs, dict), f"{workflow} declares no jobs mapping"
    return jobs


def _runs_on(workflow: str, job_key: str) -> Any:
    jobs = _jobs(workflow)
    assert job_key in jobs, f"{workflow} has no job {job_key!r} (jobs: {sorted(jobs)})"
    job = jobs[job_key]
    assert "runs-on" in job, f"{workflow}:{job_key} declares no runs-on"
    return job["runs-on"]


@pytest.mark.parametrize(("workflow", "job_key", "job_name"), MOVED_JOBS)
def test_moved_job_runs_on_the_verify_label(
    workflow: str, job_key: str, job_name: str
) -> None:
    assert _runs_on(workflow, job_key) == VERIFY_LABEL
    assert _jobs(workflow)[job_key].get("name") == job_name


@pytest.mark.parametrize(("workflow", "job_key"), STAYED_JOBS)
def test_job_left_on_the_deploy_label_stayed_there(workflow: str, job_key: str) -> None:
    """Positive control for the assertions above.

    Without this, deleting the `omnibase-deploy` label everywhere would satisfy
    every moved-job assertion while destroying the serialisation the deploy
    runner exists to provide.
    """
    assert _runs_on(workflow, job_key) == DEPLOY_LABEL


def test_no_other_job_in_the_repo_uses_the_verify_label() -> None:
    """The label is single-purpose: exactly the five jobs above, repo-wide.

    A sixth job appearing on `omnibase-verify` means the split widened without
    a decision -- the verify runner is one container, so an unreviewed addition
    reintroduces on it exactly the starvation this ticket removed from the
    deploy label.
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(WORKFLOWS.glob("*.yml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        jobs = loaded.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_key, job in jobs.items():
            if isinstance(job, dict) and job.get("runs-on") == VERIFY_LABEL:
                found.add((path.name, job_key))

    assert found == {(wf, key) for wf, key, _ in MOVED_JOBS}


def test_every_moved_job_is_host_scoped_not_merely_class_scoped() -> None:
    """The assertion the OMN-18408 acceptance criteria could not have written.

    Their falsifier names `[self-hosted, omnibase-verify]`, written before a
    second verify-class runner existed anywhere. Bare class scoping became
    unsafe on 2026-09-16, when one came online on another host carrying
    `self-hosted,omnibase-verify,arch-arm64`. Dropping `host-201` from any of
    these five would not fail to schedule -- it would schedule somewhere that
    cannot observe the .201 lane at all, which surfaces as a lane outage rather
    than as a routing mistake. That is the reading this test exists to prevent.
    """
    for workflow, job_key, _ in MOVED_JOBS:
        runs_on = _runs_on(workflow, job_key)
        assert HOST_LABEL in runs_on, (
            f"{workflow}:{job_key} is scoped to the verify CLASS but not to a "
            "HOST; it can be placed on a verify runner that cannot see the lane"
        )
