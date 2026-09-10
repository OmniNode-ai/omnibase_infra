# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17888 AC4 — the dev-lane staleness guard must fire on the measured state.

The fixtures below are not invented. They are the live readback taken while this
guard was written, 2026-09-06T18:4xZ:

* ``docker inspect omninode-runtime`` on the ``.201`` dev lane reported
  ``org.opencontainers.image.revision=2ea74bc4de76``,
  ``com.docker.compose.project=omnibase-infra``,
  ``com.omninode.build_source=workspace``, state ``running``.
* ``gh api repos/OmniNode-ai/omnibase_infra/compare/2ea74bc4de76...116b49143e``
  returned ``status=ahead``, ``ahead_by=9``,
  ``base_commit.commit.committer.date=2026-09-06T10:13:54Z``.

So at the moment this guard landed the lane was nine commits and roughly eight
and a half hours behind ``dev``, and the guard is REQUIRED to say so. A test
suite that only proved the green path would let someone quietly widen the bounds
until the measured state passed, which is what AC5 forbids.

Hermetic: the verdict functions are driven over fixtures. No docker, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from scripts.ci.check_dev_lane_staleness import (
    DEFAULT_MAX_AGE,
    DEFAULT_MAX_COMMITS_BEHIND,
    DEV_LANE_COMPOSE_PROJECT,
    DEV_LANE_CONTAINER,
    Divergence,
    LaneRevision,
    assert_lane_fence,
    evaluate,
    evaluate_convergence,
    normalize_revision,
    revisions_match,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
STALENESS_WORKFLOW = WORKFLOWS / "dev-lane-staleness.yml"
TRIGGER_WORKFLOW = WORKFLOWS / "runtime-rebuild-trigger.yml"

# --- the live readback, 2026-09-06T18:4xZ ------------------------------------
MEASURED_REVISION = "2ea74bc4de76"
MEASURED_DEV_HEAD = "116b49143eacd65ed95c0eeb7bb9dc483458d466"
MEASURED_COMMITS_BEHIND = 9
MEASURED_IMAGE_COMMITTED_AT = datetime(2026, 9, 6, 10, 13, 54, tzinfo=UTC)
NOW = datetime(2026, 9, 6, 18, 45, tzinfo=UTC)

DEV_LANE = LaneRevision(
    revision=MEASURED_REVISION,
    compose_project=DEV_LANE_COMPOSE_PROJECT,
    build_source="workspace",
    state="running",
)


def _divergence(
    status: str = "ahead",
    commits_behind: int = MEASURED_COMMITS_BEHIND,
    base_sha: str = MEASURED_REVISION,
    committed_at: datetime = MEASURED_IMAGE_COMMITTED_AT,
) -> Divergence:
    return Divergence(
        status=status,
        commits_behind=commits_behind,
        head_sha=MEASURED_DEV_HEAD,
        base_sha=base_sha,
        base_committed_at=committed_at,
    )


def _evaluate(
    lane: LaneRevision = DEV_LANE,
    divergence: Divergence | None = None,
    now: datetime = NOW,
    max_commits_behind: int = DEFAULT_MAX_COMMITS_BEHIND,
    max_age: timedelta = DEFAULT_MAX_AGE,
):
    return evaluate(
        lane=lane,
        divergence=divergence if divergence is not None else _divergence(),
        now=now,
        max_commits_behind=max_commits_behind,
        max_age=max_age,
    )


class TestTheMeasuredState:
    def test_the_live_2026_09_06_lane_state_is_reported_stale(self) -> None:
        """The state at landing time. If this ever passes, the guard is broken."""
        verdict = _evaluate()
        assert not verdict.ok
        assert [f.code for f in verdict.findings] == ["LANE_STALE"]
        detail = verdict.findings[0].detail
        assert "9 commits behind" in detail
        assert MEASURED_REVISION[:12] in detail
        assert MEASURED_DEV_HEAD[:12] in detail

    def test_the_finding_names_both_breached_bounds_not_just_one(self) -> None:
        """Nine commits AND eight hours. Reporting one hides half the divergence."""
        detail = _evaluate().findings[0].detail
        assert "commits behind" in detail
        assert "old (bound" in detail

    def test_the_finding_refuses_widening_as_the_remedy(self) -> None:
        """AC5: a green no-op would hide the finding."""
        assert "Do NOT widen the bounds" in _evaluate().findings[0].detail


class TestTheBoundsHold:
    def test_identical_is_fresh(self) -> None:
        verdict = _evaluate(
            divergence=_divergence(
                status="identical",
                commits_behind=0,
                base_sha=MEASURED_DEV_HEAD,
                committed_at=NOW - timedelta(minutes=5),
            )
        )
        assert verdict.ok
        assert "== dev head" in verdict.notes[0]

    def test_inside_both_bounds_is_fresh(self) -> None:
        verdict = _evaluate(
            divergence=_divergence(
                commits_behind=DEFAULT_MAX_COMMITS_BEHIND,
                committed_at=NOW - timedelta(minutes=90),
            )
        )
        assert verdict.ok

    def test_one_commit_past_the_commit_bound_is_stale(self) -> None:
        verdict = _evaluate(
            divergence=_divergence(
                commits_behind=DEFAULT_MAX_COMMITS_BEHIND + 1,
                committed_at=NOW - timedelta(minutes=1),
            )
        )
        assert not verdict.ok
        assert "4 commits behind (bound 3)" in verdict.findings[0].detail

    def test_one_minute_past_the_age_bound_is_stale(self) -> None:
        """Age alone is a finding — a lane can be one big merge behind."""
        verdict = _evaluate(
            divergence=_divergence(
                commits_behind=1,
                committed_at=NOW - DEFAULT_MAX_AGE - timedelta(minutes=1),
            )
        )
        assert not verdict.ok
        assert "old (bound 2h00m)" in verdict.findings[0].detail
        assert "commits behind (bound" not in verdict.findings[0].detail


class TestFailClosed:
    def test_diverged_history_is_a_finding_not_a_commit_count(self) -> None:
        verdict = _evaluate(divergence=_divergence(status="diverged"))
        assert [f.code for f in verdict.findings] == ["DIVERGED"]

    def test_a_lane_ahead_of_dev_is_a_finding(self) -> None:
        verdict = _evaluate(divergence=_divergence(status="behind"))
        assert [f.code for f in verdict.findings] == ["LANE_AHEAD_OF_BRANCH"]

    def test_an_uninterpretable_compare_status_fails_closed(self) -> None:
        verdict = _evaluate(divergence=_divergence(status="something_new"))
        assert [f.code for f in verdict.findings] == ["UNKNOWN_COMPARE_STATUS"]

    def test_a_stopped_lane_is_a_finding_even_when_the_sha_is_current(self) -> None:
        verdict = _evaluate(
            lane=LaneRevision(
                revision=MEASURED_DEV_HEAD,
                compose_project=DEV_LANE_COMPOSE_PROJECT,
                build_source="workspace",
                state="exited",
            ),
            divergence=_divergence(
                status="identical", commits_behind=0, base_sha=MEASURED_DEV_HEAD
            ),
        )
        assert not verdict.ok
        assert verdict.findings[0].code == "LANE_NOT_RUNNING"

    @pytest.mark.parametrize("raw", ["", "   ", "unknown", "none", "dev", "HEAD"])
    def test_a_sentinel_revision_label_raises_rather_than_comparing(
        self, raw: str
    ) -> None:
        with pytest.raises(ValueError, match="blank or sentinel"):
            normalize_revision(raw)

    def test_a_non_sha_revision_label_raises(self) -> None:
        with pytest.raises(ValueError, match="not a git SHA"):
            normalize_revision("v0.38.19")


class TestTheLaneFence:
    """The guard must never read a governed lane, even read-only."""

    @pytest.mark.parametrize(
        "project",
        [
            "omnibase-infra-prod",
            "omnibase-infra-stability-test",
            "omnibase-infra-judge",
            "omnibase-infra-lakshman",
        ],
    )
    def test_a_governed_lane_project_aborts(self, project: str) -> None:
        lane = LaneRevision(
            revision=MEASURED_REVISION,
            compose_project=project,
            build_source="workspace",
            state="running",
        )
        with pytest.raises(ValueError, match="lane fence"):
            assert_lane_fence(lane, DEV_LANE_CONTAINER, DEV_LANE_COMPOSE_PROJECT)

    def test_the_dev_lane_project_passes(self) -> None:
        assert_lane_fence(DEV_LANE, DEV_LANE_CONTAINER, DEV_LANE_COMPOSE_PROJECT)


class TestConvergenceMode:
    """The post-merge shape: published, but did the lane actually apply it?"""

    def test_the_measured_delivered_but_not_applied_shape_is_a_finding(self) -> None:
        """Five commands delivered and consumed on 2026-09-06; lane unchanged."""
        verdict = evaluate_convergence(
            lane=DEV_LANE,
            expected_revision=MEASURED_DEV_HEAD,
            waited=timedelta(minutes=25),
            wait_timeout=timedelta(minutes=25),
        )
        assert not verdict.ok
        assert verdict.findings[0].code == "NOT_CONVERGED"
        assert "delivered-but-not-applied" in verdict.findings[0].detail
        assert "onex.dlq.omnibase-infra.omnimarket.v1" in verdict.findings[0].detail

    def test_convergence_onto_the_merge_sha_is_clean(self) -> None:
        verdict = evaluate_convergence(
            lane=LaneRevision(
                revision=MEASURED_DEV_HEAD,
                compose_project=DEV_LANE_COMPOSE_PROJECT,
                build_source="workspace",
                state="running",
            ),
            expected_revision=MEASURED_DEV_HEAD,
            waited=timedelta(minutes=6),
            wait_timeout=timedelta(minutes=25),
        )
        assert verdict.ok

    def test_an_abbreviated_label_still_matches_the_full_merge_sha(self) -> None:
        """Some build paths stamp a short SHA; strict equality would never converge."""
        assert revisions_match(MEASURED_DEV_HEAD[:12], MEASURED_DEV_HEAD)
        assert revisions_match(MEASURED_DEV_HEAD, MEASURED_DEV_HEAD[:12])

    def test_a_different_sha_does_not_match(self) -> None:
        assert not revisions_match(MEASURED_REVISION, MEASURED_DEV_HEAD)

    def test_an_empty_revision_never_matches(self) -> None:
        assert not revisions_match("", MEASURED_DEV_HEAD)
        assert not revisions_match(MEASURED_DEV_HEAD, "")


class TestTheWiring:
    """Detection that is not wired is advisory and gets ignored (Operating Rule 5)."""

    def test_the_hourly_workflow_exists_and_is_scheduled(self) -> None:
        document = yaml.safe_load(STALENESS_WORKFLOW.read_text(encoding="utf-8"))
        triggers = document.get("on", document.get(True))
        assert "schedule" in triggers, "an unscheduled watcher watches nothing"
        assert triggers["schedule"], "empty schedule"

    def test_the_hourly_workflow_runs_where_the_lane_is(self) -> None:
        """A hosted runner cannot see the .201 docker daemon; it could only ever
        report 'I cannot look', which this guard treats as a failure."""
        document = yaml.safe_load(STALENESS_WORKFLOW.read_text(encoding="utf-8"))
        runs_on = document["jobs"]["dev-lane-staleness"]["runs-on"]
        assert "self-hosted" in runs_on

    def test_the_hourly_workflow_invokes_the_real_guard(self) -> None:
        body = STALENESS_WORKFLOW.read_text(encoding="utf-8")
        assert "scripts/ci/check_dev_lane_staleness.py" in body

    def test_the_hourly_workflow_runs_its_own_positive_control(self) -> None:
        """A zero-finding result is only evidence once the guard is shown to fire."""
        body = STALENESS_WORKFLOW.read_text(encoding="utf-8")
        assert "--positive-control" in body

    def test_the_post_merge_workflow_gained_a_convergence_job(self) -> None:
        document = yaml.safe_load(TRIGGER_WORKFLOW.read_text(encoding="utf-8"))
        jobs = document["jobs"]
        assert "verify-lane-converged" in jobs, (
            "AC4 requires a delivered-but-not-applied redeploy to surface on the "
            "same run that published it"
        )
        job = jobs["verify-lane-converged"]
        assert job["needs"] == "trigger-rebuild"
        assert "scripts/ci/check_dev_lane_staleness.py" in yaml.dump(job)
        assert "--expect-revision" in yaml.dump(job)

    def test_the_convergence_job_only_runs_when_a_command_was_published(self) -> None:
        document = yaml.safe_load(TRIGGER_WORKFLOW.read_text(encoding="utf-8"))
        condition = document["jobs"]["verify-lane-converged"]["if"]
        assert "published" in condition, (
            "a no-op trigger run has no redeploy to wait for; gating on the "
            "publisher's own output keeps the signal about delivery"
        )

    def test_the_convergence_job_checks_out_trusted_base_ref_not_fork_code(
        self,
    ) -> None:
        """This job runs on the self-hosted fleet. A merged fork PR must not put
        fork-authored code on it — check out the base branch explicitly."""
        document = yaml.safe_load(TRIGGER_WORKFLOW.read_text(encoding="utf-8"))
        steps = document["jobs"]["verify-lane-converged"]["steps"]
        checkout = next(
            s for s in steps if str(s.get("uses", "")).startswith("actions/checkout")
        )
        assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.ref }}"


class TestTheVerdictClaimsOnlyWhatTheGuardKnows:
    """The NOT_CONVERGED text asserted a fact the guard cannot observe.

    It read, verbatim: "The command was published; the lane did not apply it."

    What CI publishes is a **redeploy-start** command
    (``scripts/trigger_rebuild_on_merge.py`` ``TOPIC =
    "onex.cmd.omnimarket.redeploy-start.v1"``), and that is the only thing the
    job's ``published`` output attests. The deploy agent consumes a different
    topic, ``onex.cmd.deploy.rebuild-requested.v1``, which only
    ``node_redeploy``'s deploy effect emits. Between the two sits an
    orchestrator and that effect. So "the command was published" is true of the
    START command and says nothing about whether a rebuild command ever reached
    the agent.

    On 2026-09-10 it demonstrably had not. Measured by the peer lane that owns
    the agent, reading the control topic offsets 95..125 against the agent's own
    job store: pull request 3383 merged 07:37:40Z, and the first real
    rebuild-requested command appeared at 11:08:23Z -- three and a half hours
    after this guard's window had already closed. Publish-to-accept, once a
    command existed, was 268 milliseconds. Nothing was slow on the agent side and
    nothing was wrong with the clone. The effect is serial: it publishes one
    command, then polls until its own eleven-minute timeout, and it was working
    through five stale correlations ahead of the real ones.

    The sentence is therefore the same defect class as the two probe defects
    this ticket already fixed -- a comment asserting a premise the code cannot
    know -- and it is more costly than either, because it is the first line a
    lane reads when diagnosing a red convergence job. It sent two separate lanes
    toward the agent's clone and toward the wall-clock bound, and the cause was
    in neither.
    """

    @staticmethod
    def _detail() -> str:
        verdict = evaluate_convergence(
            lane=DEV_LANE,
            expected_revision=MEASURED_DEV_HEAD,
            waited=timedelta(minutes=25),
            wait_timeout=timedelta(minutes=25),
        )
        return verdict.findings[0].detail

    def test_it_does_not_assert_the_rebuild_command_reached_the_agent(self) -> None:
        detail = self._detail()
        assert "The command was published; the lane did not apply it." not in detail, (
            "the guard observes neither half of that sentence: it cannot see "
            "the agent's topic, and a start command is not a rebuild command"
        )

    def test_it_names_the_hop_that_can_swallow_the_start_command(self) -> None:
        detail = self._detail()
        assert "rebuild-requested" in detail, (
            "the reader has to be told the agent consumes a DIFFERENT topic, "
            "or they will go looking at the agent for a command that never "
            "reached it"
        )
        assert "redeploy-start" in detail
        assert "serial" in detail.lower(), (
            "the effect publishes one command then polls to its own timeout; a "
            "backlog ahead of this merge is the measured 2026-09-10 cause and "
            "is the first thing to check"
        )

    def test_it_still_names_the_dlq_and_keeps_the_finding_code(self) -> None:
        # The pre-existing guidance is additive, not replaced: a DLQ'd
        # downstream event is still a real cause of this shape (OMN-17888).
        detail = self._detail()
        assert "onex.dlq.omnibase-infra.omnimarket.v1" in detail
        assert "delivered-but-not-applied" in detail

    def test_the_workflow_does_not_derive_the_bound_as_end_to_end_latency(
        self,
    ) -> None:
        from pathlib import Path

        text = Path(".github/workflows/runtime-rebuild-trigger.yml").read_text(
            encoding="utf-8"
        )
        assert "The wait bound is derived from" not in text, (
            "660000 ms bounds ONE correlation's execution, not the queue ahead "
            "of it; deriving a wall-clock convergence bound from it is the "
            "false premise that made a 25-minute wait look sufficient"
        )
