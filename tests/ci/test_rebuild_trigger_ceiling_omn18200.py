# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 AC3 -- the publisher's ceiling must cover a cold shared-CI-env build.

THE INCIDENT. Run ``34669847808`` of ``runtime-rebuild-trigger.yml`` was cancelled
at the job's five-minute ceiling with step 7, "Set up CI Python environment",
still running at 3m56s. It never reached the classifier, so it published nothing,
and the merge it was firing for never reached the deploy agent. The immediately
preceding successful run spent 4m01s in the same step and finished with 40 seconds
of headroom, so the ceiling was not marginally tight; it was inside the noise.

WHY THE STEP IS BIMODAL, which is the part a bare number would hide. The job
resolves Python through the shared CI env, whose directory is keyed by a digest
including the lockfile and which lives inside each runner container. A change to
``uv.lock`` invalidates it on every container at once, so the first job on each
one pays the full build. A release version bump does exactly that, which is why
the two slow runs are adjacent in time. Warm is 4 seconds; cold is minutes; and
cold recurs on schedule rather than by accident.

These tests pin the fix and, more importantly, pin the SHAPE of the reasoning, so
a future edit that drops the ceiling back cannot do it without confronting the
measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"

#: The longest cold "Set up CI Python environment" this job has been OBSERVED to
#: complete, run 34669945527. The cancelled run's cost is unknown because it was
#: killed, so this is a floor on the real cold cost, never an estimate of it.
OBSERVED_COLD_STEP_SECONDS = 241


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def publisher(workflow: dict[str, Any]) -> dict[str, Any]:
    return dict(workflow["jobs"]["trigger-rebuild"])


def test_the_ceiling_covers_an_observed_cold_environment_build(
    publisher: dict[str, Any],
) -> None:
    """The regression this closes: a ceiling set as though the step were always
    warm, which cancels the run and publishes nothing."""
    ceiling_seconds = int(publisher["timeout-minutes"]) * 60
    assert ceiling_seconds > OBSERVED_COLD_STEP_SECONDS, (
        "the job ceiling does not cover a cold shared-CI-env build; run "
        "34669847808 was cancelled at 5 minutes with that step still running and "
        "the merge never reached the deploy agent"
    )
    # Margin, not just coverage. The observed cold build is a FLOOR -- the
    # cancelled run's true cost was never seen -- and cold builds contend with
    # each other across 88 containers after a lockfile change, so a ceiling that
    # merely exceeds the one completed measurement would fail the next time the
    # fleet is busy.
    assert ceiling_seconds >= OBSERVED_COLD_STEP_SECONDS * 3


def test_the_ceiling_is_still_bounded(publisher: dict[str, Any]) -> None:
    """This job publishes one command and does no long-running work. An unbounded
    or very large ceiling would hold a runner slot on a wedged broker connection
    rather than failing and being visible."""
    assert "timeout-minutes" in publisher
    assert int(publisher["timeout-minutes"]) <= 20


def test_the_publisher_still_resolves_python_through_the_shared_ci_env(
    publisher: dict[str, Any],
) -> None:
    """The fix must NOT be "opt this job out of the shared env".

    The digest keys on the install args, so a narrower set is a different env
    directory that no other job warms -- it would convert today's 4-second warm
    hit into a cold build on most runs, making the common case worse to improve
    the rare one. Measured: narrowing to the main dependency group takes the
    environment from 196 packages to 138, which is a 30% cut and nowhere near
    enough to make a cold build fast.
    """
    steps = list(publisher["steps"])
    setup = [s for s in steps if "setup-python-uv" in str(s.get("uses", ""))]
    assert len(setup) == 1, "the shared CI env action is the one Python resolver"
    with_block = dict(setup[0].get("with") or {})
    assert with_block.get("shared-env-enabled") == "true"
    assert "shared-env-install-args" not in with_block, (
        "narrowing the install args gives this job a private env digest that "
        "nothing else warms; see this module's docstring for the measurement"
    )


def test_every_job_in_the_workflow_carries_a_ceiling(
    workflow: dict[str, Any],
) -> None:
    """A job with no ceiling cannot be cancelled by one, but it also cannot be
    reasoned about: the failure mode this ticket is about is a ceiling nobody had
    matched to a measurement, and an absent one is the same defect unbounded."""
    for name, job in workflow["jobs"].items():
        assert "timeout-minutes" in job, name
        assert int(job["timeout-minutes"]) > 0, name
