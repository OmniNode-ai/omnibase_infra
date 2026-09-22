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
# acceptance criteria and in the authorising operator-consent row, plus the
# five OMN-18602 moved when it took the follow-up decision OMN-18408 deferred.
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
    # --- OMN-18602 -------------------------------------------------------
    # The remainder OMN-18408 left on the deploy label. Every one of them is a
    # read-only probe: none writes to the lane, and none needs the private
    # OMNI_HOME clone tree that is the deploy runner's reason to exist. They
    # were measured, not merely reasoned about -- `verify-lane-converged`
    # queued a median 29.1 min, p90 75.8, max 101.3 over its 38 non-skipped
    # runs between 2026-09-15T22:49Z and 2026-09-17T15:39Z, holding the single
    # deploy runner a median 25.0 min per run while it polled.
    (
        "runtime-rebuild-trigger.yml",
        "verify-lane-converged",
        "Verify dev lane applied the redeploy",
    ),
    (
        "runtime-rebuild-trigger-reusable.yml",
        "verify-sibling-converged",
        "Verify the dev lane vendors the merged sibling revision",
    ),
    (
        "onex-api-lab-delivery-reusable.yml",
        "verify-onex-api-delivered",
        "Verify the dev lane runs onex-api at this commit",
    ),
    ("baselines-scheduler.yml", "baselines-compute", "Baselines Batch Compute"),
    ("dlq-depth-monitor.yml", "dlq-depth-monitor", "DLQ Depth Monitor (read-only)"),
)

# Jobs that were BORN on the verify label rather than moved onto it. Kept in a
# separate tuple on purpose: MOVED_JOBS above is the record of two specific
# decisions (OMN-18408 and its OMN-18602 follow-up), and folding a new job into
# it would quietly rewrite what those decisions covered.
#
# The bar for landing here is the same one OMN-18602 settled the label's meaning
# on: the job READS the .201 lane and never mutates it. `refresh` collects a
# docker inventory through the socket the runner already mounts and opens a pull
# request; the only thing it writes is a branch in this repository.
#
# It needs the HOST label for the same reason all ten above do, and more sharply:
# a census collected against the wrong daemon is not an outage, it is a WRONG
# ANSWER committed to the repository as the documented lane topology. The
# `omnibase-ci` pool cannot be used for it at all -- 60 of its runners carry no
# host label (read live from the org runner census on 2026-09-17), so placement
# there is unpinned by construction.
NEW_VERIFY_JOBS = (
    (
        "lane-census-refresh.yml",
        "refresh",
        "Collect the lab census and open a bump PR when it has moved",
    ),
    # OMN-19175: the C11 negative-paths producer. Born on the label, and it
    # clears the OMN-18602 bar by a wide margin: four HTTP GETs against the
    # lane's onex-api plus one POST that request validation refuses BEFORE the
    # endpoint function runs, so the mutating path is structurally unreachable
    # rather than merely unused. It publishes nothing to the bus, unlike the
    # chain canary beside it.
    #
    # It needs the HOST label for the identical reason every entry above does,
    # and the failure without it is the bad kind: `omnibase-verify` names a
    # runner CLASS and a second verify-class runner exists on another lab host,
    # so a run placed there cannot see this lane at all and would report it
    # unreachable. A permanently-red probe is a disabled probe. The pair fails
    # SAFE -- if the .201 verify runner is down the job queues rather than
    # answering from somewhere blind.
    (
        "chain-canary-c11-negative-paths.yml",
        "c11-negative-paths",
        "C11 negative paths (dev lane)",
    ),
    # OMN-19195: the C12 provider-catalogue producer. Born on the label, and
    # it clears the OMN-18602 bar the same way `verify-lane-converged` does: it
    # reads the lane through the docker socket this runner already mounts. Its
    # one `docker exec` runs a Python process as the container's own
    # unprivileged user that imports the deployed package and calls pure
    # functions on in-memory copies. No file is written, no route is called and
    # nothing is published. It calls no HTTP surface at all.
    #
    # It needs the HOST label for the identical reason: on another lab host's
    # daemon there is no `onex-api` container to exec into, so an unpinned run
    # would exit 2 on every tick and be read as a lane outage.
    (
        "chain-canary-c12-provider-catalogue.yml",
        "c12-provider-catalogue",
        "C12 provider catalogue (dev lane)",
    ),
)

# Every job legally on the label, however it got there.
VERIFY_JOBS = MOVED_JOBS + NEW_VERIFY_JOBS

# Deliberately NOT moved, and after OMN-18602 this set is exactly the jobs that
# MUTATE the lane. That is the whole of what `omnibase-deploy` now means.
#
# This is the positive control that matters most in the file. `omnibase-deploy`
# is a single physical runner, and that is not an accident of provisioning --
# it is the serialisation guard for release-train tag-cut and lane refresh
# (CLAUDE.md rule 2a/12, memory feedback_serialize_same_lane_redeploys). It is
# also the only guard available: a GitHub `concurrency` group is scoped to ONE
# repository, and `verify-sibling-converged` is called from omnimarket, so no
# group could ever serialise these two against it. Draining the label of these
# jobs, or adding a second runner to it, removes the guard rather than
# relieving a queue.
STAYED_JOBS = (
    ("release-train-lab.yml", "deploy"),
    ("release-train-lab.yml", "cut-tag"),
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


@pytest.mark.parametrize(("workflow", "job_key", "job_name"), VERIFY_JOBS)
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

    assert found == {(wf, key) for wf, key, _ in VERIFY_JOBS}


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
    for workflow, job_key, _ in VERIFY_JOBS:
        runs_on = _runs_on(workflow, job_key)
        assert HOST_LABEL in runs_on, (
            f"{workflow}:{job_key} is scoped to the verify CLASS but not to a "
            "HOST; it can be placed on a verify runner that cannot see the lane"
        )
