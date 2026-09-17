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

from scripts.ci import check_dev_lane_staleness as staleness_module
from scripts.ci.check_dev_lane_staleness import (
    DEFAULT_MAX_AGE,
    DEFAULT_MAX_COMMITS_BEHIND,
    DEV_LANE_COMPOSE_PROJECT,
    DEV_LANE_CONTAINER,
    RELATION_ANCESTOR,
    RELATION_DESCENDANT,
    RELATION_IDENTICAL,
    RELATION_UNRELATED,
    Ancestry,
    Divergence,
    LaneRevision,
    assert_lane_fence,
    convergence_evidence,
    evaluate,
    evaluate_convergence,
    normalize_revision,
    parse_ancestry,
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
            # 2ea74bc4de76 is nine commits BEHIND 116b4914 on dev: an ancestor,
            # which is the stale direction and stays a finding (OMN-18388).
            ancestry=Ancestry(
                relation=RELATION_ANCESTOR,
                commits_ahead=0,
                observed_on_branch=True,
                branch="dev",
            ),
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
            # Byte-equal: the ancestry probe is never reached, so an
            # unresolvable ancestry must not make an exact match fail.
            ancestry=None,
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
            ancestry=Ancestry(
                relation=RELATION_ANCESTOR,
                commits_ahead=0,
                observed_on_branch=True,
                branch="dev",
            ),
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


# --- OMN-18388: a lane running a DESCENDANT has exercised the change ---------
#
# The live readback, 2026-09-15. Run 34934096166 (omnibase_infra#3549, merge sha
# 18b539f0) waited its full 25-minute window and failed NOT_CONVERGED at
# 08:40:32Z, so the compose-dev lab-pass receipt for that sha was emitted FAIL on
# `deployed_revision`. The .201 dev lane had in fact been rebuilt at 08:15Z to
# 0d0250a6 (#3568) -- a commit that CONTAINS 18b539f0. Byte equality reported a
# lane that had already run the change as one that never applied it.
#
# Every compare payload below is GitHub's own, captured 2026-09-15:
#   compare/18b539f0...0d0250a6  -> {"status":"ahead","ahead_by":6,"behind_by":0}
#   compare/0d0250a6...dev       -> {"status":"identical","ahead_by":0}
#   compare/0d0250a6...18b539f0  -> {"status":"behind","ahead_by":0,"behind_by":6}
#   compare/18b539f0...3ad9b3af  -> {"status":"ahead","ahead_by":7}   (PR 3569 head)
#   compare/3ad9b3af...dev       -> {"status":"behind","behind_by":1}
MERGE_SHA_18388 = "18b539f0f76fc4c5df1966d34ca4a2ebeba51f8f"
LANE_DESCENDANT_18388 = "0d0250a6c31f0ea7a3bcbeb1f2b6d2efb73cd525"
DESCENDANT_COMMITS_AHEAD = 6
# PR 3569's head: a descendant of the merge sha that dev does NOT contain.
BRANCH_BUILD_18388 = "3ad9b3af9cb8eaee8f8b01896e4a86b30137ee7d"


def _lane(revision: str) -> LaneRevision:
    return LaneRevision(
        revision=revision,
        compose_project=DEV_LANE_COMPOSE_PROJECT,
        build_source="workspace",
        state="running",
    )


def _ancestry(
    relation: str = RELATION_DESCENDANT,
    commits_ahead: int = DESCENDANT_COMMITS_AHEAD,
    observed_on_branch: bool = True,
) -> Ancestry:
    return Ancestry(
        relation=relation,
        commits_ahead=commits_ahead,
        observed_on_branch=observed_on_branch,
        branch="dev",
    )


def _converge(
    revision: str = LANE_DESCENDANT_18388,
    ancestry: Ancestry | None = None,
    waited: timedelta = timedelta(minutes=6),
):
    return evaluate_convergence(
        lane=_lane(revision),
        expected_revision=MERGE_SHA_18388,
        waited=waited,
        wait_timeout=timedelta(minutes=25),
        ancestry=ancestry if ancestry is not None else _ancestry(),
    )


class TestConvergenceIsContainmentNotByteEquality:
    """AC1/AC4 -- the real 2026-09-15 pair, in both directions."""

    def test_the_measured_descendant_lane_converges(self) -> None:
        assert _converge().ok, (
            "0d0250a6 contains 18b539f0, so the lane has run the merged change; "
            "byte equality reported this as delivered-but-not-applied and the "
            "receipt for 18b539f0 could never be a PASS"
        )

    def test_the_reverse_direction_stays_a_finding(self) -> None:
        """A lane on an ANCESTOR is the OMN-18284 stale-lane class, not this one."""
        verdict = evaluate_convergence(
            lane=_lane(MEASURED_REVISION),
            expected_revision=MERGE_SHA_18388,
            waited=timedelta(minutes=25),
            wait_timeout=timedelta(minutes=25),
            ancestry=_ancestry(relation=RELATION_ANCESTOR, commits_ahead=0),
        )
        assert not verdict.ok
        assert verdict.findings[0].code == "NOT_CONVERGED"

    def test_an_identical_relation_converges(self) -> None:
        assert _converge(ancestry=_ancestry(relation=RELATION_IDENTICAL)).ok

    def test_the_real_compare_payloads_project_onto_a_descendant(self) -> None:
        """AC4 -- GitHub's own bytes, not a shape a test author imagined."""
        ancestry = parse_ancestry(
            relation_payload={"status": "ahead", "ahead_by": 6, "behind_by": 0},
            containment_payload={"status": "identical", "ahead_by": 0},
            branch="dev",
        )
        assert ancestry.relation == RELATION_DESCENDANT
        assert ancestry.commits_ahead == DESCENDANT_COMMITS_AHEAD
        assert ancestry.observed_on_branch

    def test_the_real_compare_payloads_project_onto_an_ancestor(self) -> None:
        ancestry = parse_ancestry(
            relation_payload={"status": "behind", "ahead_by": 0, "behind_by": 6},
            containment_payload={"status": "ahead", "ahead_by": 6},
            branch="dev",
        )
        assert ancestry.relation == RELATION_ANCESTOR
        assert ancestry.observed_on_branch

    def test_an_uninterpretable_compare_status_raises(self) -> None:
        """Fail closed: a status this guard does not know is not 'converged'."""
        with pytest.raises(ValueError):
            parse_ancestry(
                relation_payload={"status": "sideways", "ahead_by": 0},
                containment_payload={"status": "identical", "ahead_by": 0},
                branch="dev",
            )


class TestOffBranchRevisionsNeverConverge:
    """AC3 -- containment in origin/dev is checked independently of the relation."""

    def test_the_real_branch_build_is_not_converged(self) -> None:
        """PR 3569's head is a DESCENDANT of the merge sha and is NOT on dev.

        This is the fixture that proves the two probes are independent: a guard
        that only asked "does the observed revision contain the merge sha" would
        report this hand-built branch image as a converged lane.
        """
        verdict = _converge(
            revision=BRANCH_BUILD_18388,
            ancestry=_ancestry(commits_ahead=7, observed_on_branch=False),
        )
        assert not verdict.ok
        assert verdict.findings[0].code == "LANE_OFF_BRANCH"

    def test_the_real_branch_build_payloads_project_as_off_branch(self) -> None:
        ancestry = parse_ancestry(
            relation_payload={"status": "ahead", "ahead_by": 7, "behind_by": 0},
            containment_payload={"status": "behind", "ahead_by": 0, "behind_by": 1},
            branch="dev",
        )
        assert ancestry.relation == RELATION_DESCENDANT
        assert not ancestry.observed_on_branch

    def test_an_unrelated_revision_is_not_converged(self) -> None:
        verdict = _converge(
            ancestry=_ancestry(
                relation=RELATION_UNRELATED, commits_ahead=0, observed_on_branch=False
            )
        )
        assert not verdict.ok
        assert verdict.findings[0].code == "LANE_OFF_BRANCH"

    def test_an_unrelated_on_branch_revision_is_not_called_older(self) -> None:
        verdict = _converge(
            ancestry=_ancestry(
                relation=RELATION_UNRELATED, commits_ahead=0, observed_on_branch=True
            )
        )
        assert not verdict.ok
        assert verdict.findings[0].code == "NOT_CONVERGED"
        assert "OLDER than the merge" not in verdict.findings[0].detail
        assert "divergent" in verdict.findings[0].detail

    def test_a_diverged_containment_status_is_off_branch(self) -> None:
        ancestry = parse_ancestry(
            relation_payload={"status": "diverged", "ahead_by": 2, "behind_by": 3},
            containment_payload={"status": "diverged", "ahead_by": 1, "behind_by": 4},
            branch="dev",
        )
        assert ancestry.relation == RELATION_UNRELATED
        assert not ancestry.observed_on_branch

    def test_an_unresolvable_ancestry_fails_closed(self) -> None:
        """An ancestry the guard could not read is not evidence of convergence."""
        verdict = evaluate_convergence(
            lane=_lane(LANE_DESCENDANT_18388),
            expected_revision=MERGE_SHA_18388,
            waited=timedelta(minutes=25),
            wait_timeout=timedelta(minutes=25),
            ancestry=None,
        )
        assert not verdict.ok
        assert verdict.findings[0].code == "ANCESTRY_UNPROVABLE"


class TestTheConvergenceEvidence:
    """AC2 -- the evidence names both shas and the relation, on one line."""

    def test_the_descendant_evidence_names_both_shas_and_the_relation(self) -> None:
        evidence = convergence_evidence(
            lane=_lane(LANE_DESCENDANT_18388),
            expected_revision=MERGE_SHA_18388,
            ancestry=_ancestry(),
            waited=timedelta(minutes=6),
            converged=True,
        )
        assert LANE_DESCENDANT_18388[:12] in evidence
        assert MERGE_SHA_18388[:12] in evidence
        assert "contains" in evidence
        assert "6 commit" in evidence
        assert "dev" in evidence

    def test_the_evidence_is_one_line_and_shell_safe(self) -> None:
        """It is written to GITHUB_OUTPUT and re-read as a receipt check field."""
        for ancestry in (
            _ancestry(),
            _ancestry(relation=RELATION_ANCESTOR, commits_ahead=0),
            _ancestry(observed_on_branch=False),
            None,
        ):
            evidence = convergence_evidence(
                lane=_lane(LANE_DESCENDANT_18388),
                expected_revision=MERGE_SHA_18388,
                ancestry=ancestry,
                waited=timedelta(minutes=6),
                converged=False,
            )
            assert evidence, "an empty evidence string is refused by the receipt model"
            assert "\n" not in evidence and "\r" not in evidence
            assert '"' not in evidence and "`" not in evidence and "$" not in evidence

    def test_the_off_branch_evidence_says_so(self) -> None:
        evidence = convergence_evidence(
            lane=_lane(BRANCH_BUILD_18388),
            expected_revision=MERGE_SHA_18388,
            ancestry=_ancestry(commits_ahead=7, observed_on_branch=False),
            waited=timedelta(minutes=25),
            converged=False,
        )
        assert BRANCH_BUILD_18388[:12] in evidence
        assert MERGE_SHA_18388[:12] in evidence
        assert "not contained" in evidence

    def test_the_unprovable_evidence_names_the_expected_sha(self) -> None:
        evidence = convergence_evidence(
            lane=_lane(LANE_DESCENDANT_18388),
            expected_revision=MERGE_SHA_18388,
            ancestry=None,
            waited=timedelta(minutes=25),
            converged=False,
        )
        assert MERGE_SHA_18388[:12] in evidence


class TestTheReceiptCarriesTheAncestryEvidence:
    """AC2's wiring half: the receipt's sha stays exact, its evidence is live."""

    @staticmethod
    def _verify_job() -> dict:
        document = yaml.safe_load(TRIGGER_WORKFLOW.read_text(encoding="utf-8"))
        return document["jobs"]["verify-lane-converged"]

    def test_the_receipt_sha_is_the_unmodified_merge_sha(self) -> None:
        emit = next(
            s
            for s in self._verify_job()["steps"]
            if "lab_pass_receipt.py emit" in str(s.get("run", ""))
        )
        assert '--sha "$MERGE_SHA"' in emit["run"], (
            "rule 24(b) keys the artifact by the exact merge sha; a descendant "
            "window belongs in the CHECK, never in the key"
        )

    def test_the_deployed_revision_evidence_comes_from_the_converge_step(self) -> None:
        emit = next(
            s
            for s in self._verify_job()["steps"]
            if "lab_pass_receipt.py emit" in str(s.get("run", ""))
        )
        assert "steps.converge.outputs.evidence" in yaml.dump(emit), (
            "a static evidence string cannot name the revision the lane was "
            "actually observed at, which is what AC2 requires"
        )

    def test_the_evidence_is_passed_through_env_not_interpolated_into_the_shell(
        self,
    ) -> None:
        emit = next(
            s
            for s in self._verify_job()["steps"]
            if "lab_pass_receipt.py emit" in str(s.get("run", ""))
        )
        assert "steps.converge.outputs.evidence" not in str(emit["run"]), (
            "expression-interpolating a value into a run block is the script "
            "injection shape; pass it through env and dereference the variable"
        )

    def test_cancelled_or_skipped_convergence_is_recorded_as_inconclusive(
        self,
    ) -> None:
        emit = next(
            s
            for s in self._verify_job()["steps"]
            if "lab_pass_receipt.py emit" in str(s.get("run", ""))
        )
        body = emit["run"]
        assert "CONVERGE_OUTCOME" in yaml.dump(emit)
        # OMN-18573. A guard that ended without writing its own verdict has
        # established nothing about the LANE, so the fallback records
        # INDETERMINATE rather than FAIL. It is still not a pass -- the receipt
        # stays non-PASS and rule 24(b) still refuses the sha -- and the
        # evidence still names the outcome the step ended on, which is the
        # property this test has always been about.
        assert "the convergence guard ended ${CONVERGE_OUTCOME}" in body
        assert "DEPLOYED_REVISION_VERDICT=indeterminate" in body
        assert "DEPLOYED_REVISION_VERDICT=ok" not in body.split("if [[")[0]
        assert "steps.converge.outcome == 'success' && 'ok' || 'fail'" not in yaml.dump(
            emit
        )

    def test_fallback_evidence_uses_a_validated_merge_sha_copy(self) -> None:
        emit = next(
            s
            for s in self._verify_job()["steps"]
            if "lab_pass_receipt.py emit" in str(s.get("run", ""))
        )
        body = emit["run"]
        assert "SAFE_MERGE_SHA" in body
        assert "^[0-9a-f]{40}$" in body
        assert "--expect-revision ${SAFE_MERGE_SHA}" in body


class TestAncestryResolver:
    def test_transient_resolution_failure_is_not_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0
        resolved = _ancestry()

        def fake_read_ancestry(
            repo: str, branch: str, expected_revision: str, observed_revision: str
        ) -> Ancestry:
            nonlocal calls
            assert (repo, branch, expected_revision, observed_revision) == (
                "repo/name",
                "dev",
                MERGE_SHA_18388,
                LANE_DESCENDANT_18388,
            )
            calls += 1
            if calls == 1:
                raise RuntimeError("502")
            return resolved

        monkeypatch.setattr(staleness_module, "read_ancestry", fake_read_ancestry)
        resolver = staleness_module._AncestryResolver(
            repo="repo/name", branch="dev", expected=MERGE_SHA_18388
        )

        assert resolver.resolve(LANE_DESCENDANT_18388) is None
        assert resolver.resolve(LANE_DESCENDANT_18388) == resolved
        assert resolver.resolve(LANE_DESCENDANT_18388) == resolved
        assert calls == 2

    def test_convergence_loop_does_not_reevaluate_after_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out"))
        reads = [
            _lane(MEASURED_REVISION),
            _lane(LANE_DESCENDANT_18388),
        ]
        observed_reads: list[str] = []

        def fake_read_lane(args) -> LaneRevision:  # type: ignore[no-untyped-def]
            lane = reads.pop(0)
            observed_reads.append(lane.revision)
            return lane

        def fake_sleep(_seconds: float) -> None:
            return None

        def fake_monotonic() -> float:
            fake_monotonic.value += 1
            return fake_monotonic.value

        fake_monotonic.value = 0.0  # type: ignore[attr-defined]

        monkeypatch.setattr(staleness_module, "_read_lane", fake_read_lane)
        monkeypatch.setattr(staleness_module.time, "sleep", fake_sleep)
        monkeypatch.setattr(staleness_module.time, "monotonic", fake_monotonic)
        monkeypatch.setattr(
            staleness_module._AncestryResolver,
            "resolve",
            lambda _self, observed: _ancestry()
            if observed == LANE_DESCENDANT_18388
            else _ancestry(relation=RELATION_ANCESTOR, commits_ahead=0),
        )

        args = type(
            "Args",
            (),
            {
                "expect_revision": MERGE_SHA_18388,
                "wait_timeout": timedelta(minutes=25),
                "poll_interval": timedelta(seconds=60),
                "repo": "repo/name",
                "branch": "dev",
                # OMN-18436: the guard now also publishes the identity of the
                # container it read, so the lab-pass probe can prove its HTTP
                # reads came from the same generation. Both fields are part of
                # the function's real input surface.
                "container": "omninode-runtime",
                "deployed_revision": "",
                # OMN-18573: the convergence budget is measured from the deploy
                # agent's acceptance of this run's command, so the guard now
                # also takes the agent surface and the correlation id. Left
                # empty here on purpose -- this case is about the LOOP, and an
                # unestablished budget must not change how many times the lane
                # is read before a convergence is recognised.
                "agent_url": "",
                "correlation_id": "",
                "agent_timeout_seconds": 10.0,
                "wall_clock_seconds": 1500,
            },
        )()

        assert staleness_module._run_convergence_mode(args) == 0
        assert observed_reads == [MEASURED_REVISION, LANE_DESCENDANT_18388]
