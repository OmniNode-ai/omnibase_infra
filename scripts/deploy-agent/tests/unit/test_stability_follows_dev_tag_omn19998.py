# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: the stability lane follows the omnimarket dev tag the dev lane runs.

The lab's only dev deploy agent was fenced to dev, and every remote lane's
landing-compose guard refuses ``refresh_stability_lane.sh``, so the stability
lane had no sanctioned deploy path and ran a 13-day-old omnimarket. A dev agent
whose fence also admits the stability lane rebuilds it, on the idle tick that
follows the newest dev tag for the dev lane, onto that tag once dev runs it, as
its own job named ``agent/stability-follow-tag``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from deploy_agent.agent import DeployAgent
from deploy_agent.events import BuildSource, EnumRuntimeLane, Scope
from deploy_agent.idle_converge import (
    STABILITY_FOLLOW_REQUESTER,
    EnumStabilityFollowVerdict,
    ModelDevTag,
    ModelStabilityFollowInputs,
    decide_stability_follow,
    stability_follow_command,
)
from deploy_agent.job_state import JobStore
from deploy_agent.lane_policy import (
    follows_stability,
    is_lab_dev_fence,
    parse_allowed_lanes,
)

from .test_idle_converge_omn19509 import _agent

pytestmark = pytest.mark.unit

OLD = "1" * 40
MID = "2" * 40
NEW = "3" * 40
PAST = "4" * 40
STABILITY = EnumRuntimeLane.STABILITY_TEST.value
LAB_FENCE = f"dev,{STABILITY}"


def _tag(sha: str, hour: int) -> ModelDevTag:
    cut = datetime(2026, 10, 16, hour, 0, tzinfo=UTC)
    return ModelDevTag(name=f"dev-{cut:%Y%m%d%H%M%S}-{sha[:12]}", sha=sha, cut_at=cut)


TAGS = (_tag(OLD, 6), _tag(MID, 7), _tag(NEW, 9))


class TestLabFence:
    @pytest.mark.parametrize(
        ("raw", "lab", "follows"),
        [
            ("dev", True, False),
            (LAB_FENCE, True, True),
            (STABILITY, False, False),
            (f"{LAB_FENCE},prod", False, False),
            ("prod", False, False),
        ],
    )
    def test_the_lab_fence_is_dev_alone_or_with_stability(
        self, raw: str, lab: bool, follows: bool
    ) -> None:
        allowed = parse_allowed_lanes(raw)
        assert is_lab_dev_fence(allowed) is lab
        assert follows_stability(allowed) is follows


def _inputs(**overrides: object) -> ModelStabilityFollowInputs:
    base: dict[str, object] = {
        "tags": TAGS,
        "dev_running_ref": NEW,
        "stability_running_ref": OLD,
        "job_active": False,
    }
    base.update(overrides)
    return ModelStabilityFollowInputs.model_validate(base)


class TestDecideStabilityFollow:
    def test_follows_the_tag_dev_runs(self) -> None:
        decision = decide_stability_follow(_inputs())
        assert decision.verdict is EnumStabilityFollowVerdict.FOLLOW
        assert decision.newest_tag_sha == NEW
        assert decision.newest_tag == TAGS[-1].name

    def test_follows_when_dev_runs_a_commit_past_the_tag(self) -> None:
        decision = decide_stability_follow(
            _inputs(dev_running_ref=PAST, dev_contains_newest=True)
        )
        assert decision.verdict is EnumStabilityFollowVerdict.FOLLOW

    def test_waits_while_dev_does_not_run_the_newest_tag(self) -> None:
        decision = decide_stability_follow(_inputs(dev_running_ref=MID))
        assert decision.verdict is EnumStabilityFollowVerdict.DEV_BEHIND

    def test_an_unread_dev_lane_waits(self) -> None:
        decision = decide_stability_follow(_inputs(dev_running_ref=None))
        assert decision.verdict is EnumStabilityFollowVerdict.DEV_BEHIND

    @pytest.mark.parametrize(
        "overrides",
        [
            {"stability_running_ref": NEW},
            {"stability_running_ref": PAST, "stability_contains_newest": True},
        ],
    )
    def test_a_stability_lane_on_the_tag_is_up_to_date(
        self, overrides: dict[str, object]
    ) -> None:
        decision = decide_stability_follow(_inputs(**overrides))
        assert decision.verdict is EnumStabilityFollowVerdict.UP_TO_DATE

    def test_an_unread_stability_lane_is_never_rebuilt_on_a_guess(self) -> None:
        decision = decide_stability_follow(_inputs(stability_running_ref=None))
        assert decision.verdict is EnumStabilityFollowVerdict.RUNNING_UNREADABLE

    def test_one_attempt_per_tag(self) -> None:
        decision = decide_stability_follow(_inputs(attempted=frozenset({NEW})))
        assert decision.verdict is EnumStabilityFollowVerdict.ALREADY_ATTEMPTED

    @pytest.mark.parametrize("tags", [None, ()])
    def test_no_tag_read_refuses(self, tags: object) -> None:
        decision = decide_stability_follow(_inputs(tags=tags))
        assert decision.verdict is EnumStabilityFollowVerdict.NO_TAG

    def test_a_job_in_flight_waits(self) -> None:
        decision = decide_stability_follow(_inputs(job_active=True))
        assert decision.verdict is EnumStabilityFollowVerdict.JOB_ACTIVE

    def test_a_frozen_instance_does_not_follow(self) -> None:
        decision = decide_stability_follow(_inputs(frozen_reason="frozen by flag"))
        assert decision.verdict is EnumStabilityFollowVerdict.FROZEN


def test_the_follow_command_is_a_workspace_stability_rebuild() -> None:
    cmd = stability_follow_command()
    assert cmd.requested_by == STABILITY_FOLLOW_REQUESTER
    assert cmd.runtime_lane is EnumRuntimeLane.STABILITY_TEST
    assert cmd.build_source is BuildSource.WORKSPACE
    assert cmd.scope is Scope.RUNTIME
    assert stability_follow_command().correlation_id != cmd.correlation_id


def _lab_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fence: str,
    dev_ref: str = NEW,
    stability_ref: str | None = OLD,
) -> DeployAgent:
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", fence)
    agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=True)
    # The dev lane runs its followed heads, so it needs no converge.
    agent._idle_read_running_ref = lambda repo: dev_ref
    agent._idle_read_head_ref = lambda repo: dev_ref
    agent._idle_read_dev_tags = lambda repo: list(TAGS)
    agent._idle_running_contains = lambda repo, ancestor, ref: False
    agent._stability_read_running_ref = lambda repo: stability_ref
    return agent


def _jobs(tmp_path: Path) -> list[dict[str, object]]:
    store = JobStore(tmp_path / "jobs")
    jobs = [store.load(UUID(p.stem)) for p in (tmp_path / "jobs").glob("*.json")]
    return [job.command for job in jobs if job is not None]


class TestStabilityFollowInTheAgent:
    def test_runs_one_stability_rebuild_once_dev_runs_the_tag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _lab_agent(tmp_path, monkeypatch, fence=LAB_FENCE)

        agent._maybe_idle_converge()

        commands = _jobs(tmp_path)
        assert len(commands) == 1
        assert commands[0]["requested_by"] == STABILITY_FOLLOW_REQUESTER
        assert commands[0]["runtime_lane"] == STABILITY
        assert "rebuild_scope" in getattr(agent.executor, "calls", [])

        # The same tag is not followed twice.
        agent._idle_converge_last_check = None
        agent._maybe_idle_converge()
        assert len(_jobs(tmp_path)) == 1

    def test_does_not_follow_while_dev_runs_an_older_tag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _lab_agent(tmp_path, monkeypatch, fence=LAB_FENCE, dev_ref=MID)

        agent._maybe_idle_converge()

        assert _jobs(tmp_path) == []
        assert (
            agent._stability_follow_last_verdict
            is EnumStabilityFollowVerdict.DEV_BEHIND
        )

    def test_a_stability_lane_on_the_tag_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _lab_agent(tmp_path, monkeypatch, fence=LAB_FENCE, stability_ref=NEW)

        agent._maybe_idle_converge()

        assert _jobs(tmp_path) == []

    @pytest.mark.parametrize("fence", ["dev", f"{LAB_FENCE},prod"])
    def test_an_agent_whose_fence_is_not_lab_dev_and_stability_never_follows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fence: str
    ) -> None:
        agent = _lab_agent(tmp_path, monkeypatch, fence=fence)

        agent._maybe_idle_converge()

        assert _jobs(tmp_path) == []
        assert agent._stability_follow_last_verdict is None
