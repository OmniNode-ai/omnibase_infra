# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: the stability lane follows the omnimarket dev tag the dev lane runs.

The lab's only dev deploy agent was fenced to dev, and every remote lane's
landing-compose guard refuses ``refresh_stability_lane.sh``, so the stability
lane had no sanctioned deploy path and ran a 13-day-old omnimarket. A dev agent
whose fence also admits the stability lane rebuilds it, on the idle tick that
follows the newest dev tag for the dev lane, onto the tag the dev lane itself
runs -- never a pinned sha, never the newest tag unless dev ran it -- and only
once the dev lane's own newest terminal deploy record is a success (OMN-21485),
as its own job named ``agent/stability-follow-tag`` staging exactly that tag
(``sibling_tags``) in place of the newest one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.idle_converge import (
    STABILITY_FOLLOW_REQUESTER,
    EnumStabilityFollowVerdict,
    ModelDevTag,
    ModelStabilityFollowInputs,
    decide_stability_follow,
    select_dev_ran_tag,
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
DEV = EnumRuntimeLane.DEV.value
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
    # OMN-21485: the dev receipt is read and passed, and the tag dev runs is
    # the newest one; every case below moves one input off that base.
    base: dict[str, object] = {
        "tags": TAGS,
        "dev_ran_tag_sha": NEW,
        "dev_receipt": "success",
        "stability_running_ref": OLD,
        "job_active": False,
    }
    base.update(overrides)
    return ModelStabilityFollowInputs.model_validate(base)


class TestDecideStabilityFollow:
    def test_follows_the_tag_dev_runs(self) -> None:
        decision = decide_stability_follow(_inputs())
        assert decision.verdict is EnumStabilityFollowVerdict.FOLLOW
        assert decision.followed_tag_sha == NEW
        assert decision.followed_tag == TAGS[-1].name

    def test_follows_the_tag_dev_ran_even_when_a_newer_tag_exists(self) -> None:
        """The starvation OMN-21485 fixes: dev lags the newest tag, and the
        stability lane follows the tag dev actually ran instead of waiting."""
        decision = decide_stability_follow(_inputs(dev_ran_tag_sha=MID))
        assert decision.verdict is EnumStabilityFollowVerdict.FOLLOW
        assert decision.followed_tag_sha == MID
        assert decision.followed_tag_sha != NEW
        assert decision.followed_tag == TAGS[1].name
        # And the follow stages exactly that tag, never the newest one.
        cmd = stability_follow_command(TAGS[1].name, "omnimarket")
        assert cmd.sibling_tags == {"omnimarket": TAGS[1].name}

    @pytest.mark.parametrize("receipt", ["failed", None])
    def test_a_dev_receipt_that_is_not_a_pass_waits(self, receipt: object) -> None:
        # The failing control: the same inputs with a passing receipt follow.
        control = decide_stability_follow(_inputs())
        assert control.verdict is EnumStabilityFollowVerdict.FOLLOW
        decision = decide_stability_follow(_inputs(dev_receipt=receipt))
        assert decision.verdict is EnumStabilityFollowVerdict.DEV_RECEIPT_NOT_PASSED
        assert ("unread" if receipt is None else "failed") in decision.detail

    @pytest.mark.parametrize("dev_ran_tag_sha", [None, PAST])
    def test_a_dev_lane_that_runs_no_dev_tag_waits(
        self, dev_ran_tag_sha: object
    ) -> None:
        decision = decide_stability_follow(_inputs(dev_ran_tag_sha=dev_ran_tag_sha))
        assert decision.verdict is EnumStabilityFollowVerdict.DEV_BEHIND

    @pytest.mark.parametrize(
        "overrides",
        [
            {"stability_running_ref": NEW},
            {"stability_running_ref": PAST, "stability_contains_tag": True},
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


class TestSelectDevRanTag:
    def test_the_tag_whose_own_sha_the_dev_lane_runs(self) -> None:
        selected = select_dev_ran_tag(TAGS, MID, lambda ancestor, ref: False)
        assert selected is TAGS[1]

    def test_a_commit_containing_the_tag_selects_that_tag(self) -> None:
        """The dev ref is past MID and contains it, and no other tag."""
        selected = select_dev_ran_tag(TAGS, PAST, lambda ancestor, ref: ancestor == MID)
        assert selected is TAGS[1]

    def test_a_ref_no_tag_matches_selects_none(self) -> None:
        assert select_dev_ran_tag(TAGS, PAST, lambda ancestor, ref: False) is None

    def test_an_unread_dev_lane_selects_none(self) -> None:
        assert select_dev_ran_tag(TAGS, None, lambda ancestor, ref: True) is None

    def test_the_walk_is_newest_first_when_several_tags_match(self) -> None:
        """Both NEW and MID are contained in the ref: the newest one wins."""
        selected = select_dev_ran_tag(TAGS, PAST, lambda ancestor, ref: True)
        assert selected is TAGS[2]


def test_the_follow_command_is_a_workspace_stability_rebuild() -> None:
    cmd = stability_follow_command(TAGS[1].name, "omnimarket")
    assert cmd.requested_by == STABILITY_FOLLOW_REQUESTER
    assert cmd.runtime_lane is EnumRuntimeLane.STABILITY_TEST
    assert cmd.build_source is BuildSource.WORKSPACE
    assert cmd.scope is Scope.RUNTIME
    assert cmd.sibling_tags == {"omnimarket": TAGS[1].name}
    assert (
        stability_follow_command(TAGS[1].name, "omnimarket").correlation_id
        != cmd.correlation_id
    )


def _rebuild_command(**overrides: object) -> ModelRebuildRequested:
    base: dict[str, object] = {
        "correlation_id": uuid4(),
        "requested_by": "test/stability-follow-tag",
        "scope": Scope.RUNTIME,
        "runtime_lane": EnumRuntimeLane.STABILITY_TEST,
    }
    base.update(overrides)
    return ModelRebuildRequested.model_validate(base)


class TestRebuildCommandSiblingTags:
    """OMN-21485: sibling_tags names the sibling repositories a lab build pins."""

    def test_a_repository_and_its_dev_tag_ride_with_the_command(self) -> None:
        cmd = _rebuild_command(sibling_tags={"omnimarket": TAGS[1].name})
        assert cmd.sibling_tags == {"omnimarket": TAGS[1].name}

    def test_a_command_that_omits_sibling_tags_defaults_to_empty(self) -> None:
        assert _rebuild_command().sibling_tags == {}

    def test_a_key_that_is_not_a_repository_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="match pattern"):
            _rebuild_command(sibling_tags={"Omni-Market": TAGS[1].name})

    def test_a_value_that_is_not_a_dev_tag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="match pattern"):
            _rebuild_command(sibling_tags={"omnimarket": "v0.4.1"})

    def test_a_value_that_is_a_full_sha_is_refused(self) -> None:
        """A pinned sha is not a tag: stability never rebuilds on a pin."""
        with pytest.raises(ValueError, match="match pattern"):
            _rebuild_command(sibling_tags={"omnimarket": NEW})


def _lab_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fence: str,
    dev_ref: str | None = NEW,
    stability_ref: str | None = OLD,
    receipt: str | None = "success",
    fake_receipt: bool = True,
    follow_setting: str = "",
) -> DeployAgent:
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", fence)
    monkeypatch.setenv("DEPLOY_AGENT_FOLLOW_STABILITY", follow_setting)
    agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=True)
    # The dev lane runs its followed heads, so it needs no converge.
    agent._idle_read_running_ref = lambda repo: dev_ref
    agent._idle_read_head_ref = lambda repo: dev_ref
    agent._idle_read_dev_tags = lambda repo: list(TAGS)
    agent._idle_running_contains = lambda repo, ancestor, ref: False
    agent._stability_read_running_ref = lambda repo: stability_ref
    # OMN-21485: the dev lane's own newest terminal deploy record.
    if fake_receipt:
        monkeypatch.setattr(
            agent.job_store, "latest_terminal_status", lambda lane: receipt
        )
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
        assert commands[0]["sibling_tags"] == {"omnimarket": TAGS[-1].name}
        assert "rebuild_scope" in getattr(agent.executor, "calls", [])

        # The same tag is not followed twice.
        agent._idle_converge_last_check = None
        agent._maybe_idle_converge()
        assert len(_jobs(tmp_path)) == 1

    def test_the_unit_shape_follows_with_the_setting_and_not_without_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The unit declares the dev fence plus the follow setting (OMN-21485).

        The failing control: the same dev fence without the setting never
        follows, so the setting alone is what admits the stability lane.
        """
        off = _lab_agent(tmp_path, monkeypatch, fence="dev")
        off._maybe_idle_converge()
        assert _jobs(tmp_path) == []

        on = _lab_agent(tmp_path, monkeypatch, fence="dev", follow_setting="1")
        on._maybe_idle_converge()
        assert len(_jobs(tmp_path)) == 1

    def test_follows_the_tag_dev_ran_even_when_a_newer_tag_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _lab_agent(tmp_path, monkeypatch, fence=LAB_FENCE, dev_ref=MID)

        agent._maybe_idle_converge()

        commands = _jobs(tmp_path)
        assert len(commands) == 1
        assert commands[0]["sibling_tags"] == {"omnimarket": TAGS[1].name}

    @pytest.mark.parametrize(
        ("dev_status", "jobs_after"), [("success", 1), ("failed", 0)]
    )
    def test_the_dev_receipt_is_read_from_the_agents_own_job_records(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        dev_status: Literal["success", "failed"],
        jobs_after: int,
    ) -> None:
        """The seam: JobStore.latest_terminal_status feeds the follow rule.

        No fake of the receipt here: a real dev-lane job record written by the
        agent's own store decides, and its failure is the failing control.
        """
        agent = _lab_agent(tmp_path, monkeypatch, fence=LAB_FENCE, fake_receipt=False)
        cid = uuid4()
        agent.job_store.accept(cid, {"scope": "runtime", "runtime_lane": DEV})
        agent.job_store.complete(cid, dev_status)
        before = len(_jobs(tmp_path))

        agent._maybe_idle_converge()

        assert len(_jobs(tmp_path)) - before == jobs_after

    def test_the_commands_sibling_tags_reach_the_executor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The seam: the follow command's tag rides into rebuild_scope."""
        agent = _lab_agent(tmp_path, monkeypatch, fence=LAB_FENCE, dev_ref=MID)
        seen: dict[str, object] = {}

        def record(*args: object, **kwargs: object) -> list[str]:
            seen.update(kwargs)
            return []

        monkeypatch.setattr(agent.executor, "rebuild_scope", record, raising=False)

        agent._maybe_idle_converge()

        assert seen["sibling_tags"] == {"omnimarket": TAGS[1].name}
        assert seen["lane"] is EnumRuntimeLane.STABILITY_TEST

    @pytest.mark.parametrize("receipt", ["failed", None])
    def test_a_dev_receipt_that_did_not_pass_creates_no_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipt: object
    ) -> None:
        agent = _lab_agent(tmp_path, monkeypatch, fence=LAB_FENCE, receipt=receipt)

        agent._maybe_idle_converge()

        assert _jobs(tmp_path) == []
        assert (
            agent._stability_follow_last_verdict
            is EnumStabilityFollowVerdict.DEV_RECEIPT_NOT_PASSED
        )

    def test_a_dev_lane_that_runs_no_dev_tag_waits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _lab_agent(tmp_path, monkeypatch, fence=LAB_FENCE, dev_ref=PAST)

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
