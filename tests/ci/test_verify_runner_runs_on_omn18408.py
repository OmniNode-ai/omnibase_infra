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

import json
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
    # Still host-pinned, knowingly (OMN-19894 residual): its AWS OIDC role trust
    # is scoped to the dev branch, so no feature-branch run can prove it off
    # this host, and the bastion's reachability from the other lab hosts is
    # unmeasured. It moves when both are settled.
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
)

# Jobs that were BORN on the host-pinned verify label. OMN-19894 moved every
# one of them to the overlay (UNPINNED_JOBS below), so none is left; the tuple
# stays so the moved-set record above keeps its meaning.
NEW_VERIFY_JOBS: tuple[tuple[str, str, str], ...] = ()

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


# OMN-19894 (operator rulings 2026-09-28T01:58:06Z and 01:58:17Z: a check must
# not rely on one machine). These jobs left the host label. They run on the
# overlay's runner pool and grade the first lane of the overlay's ordered list
# that answers; where a job must sit beside the lane's docker daemon, the
# runner labels are that lane's overlay entry, never a literal in the workflow.
POOL_RUNS_ON = "${{ fromJSON(vars.LAB_PROBE_RUNS_ON_JSON) }}"
LANE_SIDE_RUNS_ON = "${{ fromJSON(needs.resolve-lane.outputs.docker_runs_on) }}"
UNPINNED_JOBS = (
    ("chain-canary.yml", "chain-canary", POOL_RUNS_ON),
    ("chain-canary-c11-negative-paths.yml", "c11-negative-paths", POOL_RUNS_ON),
    ("chain-canary-c12-provider-catalogue.yml", "resolve-lane", POOL_RUNS_ON),
    (
        "chain-canary-c12-provider-catalogue.yml",
        "c12-provider-catalogue",
        LANE_SIDE_RUNS_ON,
    ),
    ("chain-canary-c16-receipt-identity.yml", "c16-receipt-identity", POOL_RUNS_ON),
    ("chain-canary-c28-consumer-flow.yml", "resolve-lane", POOL_RUNS_ON),
    ("chain-canary-c28-consumer-flow.yml", "c28-consumer-flow", LANE_SIDE_RUNS_ON),
    ("baselines-scheduler.yml", "baselines-compute", POOL_RUNS_ON),
    ("dlq-depth-monitor.yml", "dlq-depth-monitor", POOL_RUNS_ON),
    ("r1-front-door-probe.yml", "r1-front-door-probe", POOL_RUNS_ON),
    ("dev-lane-liveness.yml", "resolve-lane", POOL_RUNS_ON),
    ("dev-lane-liveness.yml", "dev-lane-liveness", LANE_SIDE_RUNS_ON),
    ("dev-lane-staleness.yml", "resolve-lane", POOL_RUNS_ON),
    ("dev-lane-staleness.yml", "dev-lane-staleness", LANE_SIDE_RUNS_ON),
    ("provider-rung-canary.yml", "resolve-lane", POOL_RUNS_ON),
    ("provider-rung-canary.yml", "provider-rung-canary", LANE_SIDE_RUNS_ON),
    (
        "lane-census-refresh.yml",
        "refresh",
        "${{ fromJSON(vars.LANE_CENSUS_RUNS_ON_JSON) }}",
    ),
)


@pytest.mark.parametrize(("workflow", "job_key", "runs_on"), UNPINNED_JOBS)
def test_unpinned_job_runs_on_the_overlay_not_a_host(
    workflow: str, job_key: str, runs_on: str
) -> None:
    assert _jobs(workflow)[job_key]["runs-on"] == runs_on
    assert HOST_LABEL not in (WORKFLOWS / workflow).read_text(encoding="utf-8")


#: OMN-19507 AC2: the two per-merge convergence jobs run where the deploy-agent
#: route sends the merge. Their ``runs-on`` is this expression, fed by the
#: trigger job's ``verify_runs_on`` output, which
#: ``scripts/ci/deploy_lane_verify_route.py`` resolves from
#: ``config/deploy_lane_routing.yaml``.
ROUTED_RUNS_ON = "${{ fromJSON(needs.trigger-rebuild.outputs.verify_runs_on) }}"
ROUTED_JOBS = frozenset(
    {
        ("runtime-rebuild-trigger.yml", "verify-lane-converged"),
        ("runtime-rebuild-trigger-reusable.yml", "verify-sibling-converged"),
    }
)


def _routed_labels(requester: str = "gha/omnibase_infra/pr-1") -> Any:
    """What a routed job's ``runs-on`` resolves to for `requester` under the
    COMMITTED table.

    Every requester but omnimarket still resolves to dev-201/host-201 -- the
    literal the two jobs carried before OMN-19507 -- so that stays the default
    here. Task B8 (OMN-19510) landed the real omnimarket -> dev-202 route,
    moving that one caller's own resolution to its host-scoped runner; see
    ``test_the_reusable_verify_job_moves_with_omnimarkets_own_route`` below and
    the full pin in tests/ci/test_deploy_lane_verify_route_omn19507.py.
    """
    from scripts.ci.deploy_lane_verify_route import job_outputs, load_table, resolve

    return json.loads(
        job_outputs(resolve(load_table(), runtime_lane="dev", requested_by=requester))[
            "verify_runs_on"
        ]
    )


def _runs_on(workflow: str, job_key: str) -> Any:
    jobs = _jobs(workflow)
    assert job_key in jobs, f"{workflow} has no job {job_key!r} (jobs: {sorted(jobs)})"
    job = jobs[job_key]
    assert "runs-on" in job, f"{workflow}:{job_key} declares no runs-on"
    if (workflow, job_key) in ROUTED_JOBS:
        assert job["runs-on"] == ROUTED_RUNS_ON, (workflow, job_key, job["runs-on"])
        return _routed_labels()
    return job["runs-on"]


@pytest.mark.parametrize(("workflow", "job_key", "job_name"), VERIFY_JOBS)
def test_moved_job_runs_on_the_verify_label(
    workflow: str, job_key: str, job_name: str
) -> None:
    assert _runs_on(workflow, job_key) == VERIFY_LABEL
    assert _jobs(workflow)[job_key].get("name") == job_name


def test_the_reusable_verify_job_moves_with_omnimarkets_own_route() -> None:
    """Task B8 (OMN-19510): when omnimarket itself calls the reusable
    workflow, ``verify-sibling-converged`` reads the SAME
    ``verify_runs_on`` output as every other caller, but that output now
    resolves to dev-202's own host-scoped runner rather than the dev-201
    literal every other caller still gets."""
    assert ("runtime-rebuild-trigger-reusable.yml", "verify-sibling-converged") in (
        ROUTED_JOBS
    )
    assert _routed_labels("gha/omnimarket/pr-1") == [
        "self-hosted",
        "omnibase-verify",
        "host-202",
    ]


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
            if not isinstance(job, dict):
                continue
            if job.get("runs-on") == VERIFY_LABEL or (
                job.get("runs-on") == ROUTED_RUNS_ON
                and _routed_labels() == VERIFY_LABEL
            ):
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
    from scripts.ci.deploy_lane_verify_route import load_table

    for name, spec in load_table()["instances"].items():
        labels = spec["verify"]["runner_labels"]
        assert any(label.startswith("host-") for label in labels), (
            f"routed instance {name} is scoped to the verify CLASS but not to a HOST"
        )
    for workflow, job_key, _ in VERIFY_JOBS:
        runs_on = _runs_on(workflow, job_key)
        assert HOST_LABEL in runs_on, (
            f"{workflow}:{job_key} is scoped to the verify CLASS but not to a "
            "HOST; it can be placed on a verify runner that cannot see the lane"
        )
