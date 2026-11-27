# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24562: an unhealthy stability runtime runs no revision, so the follow rebuilds it.

A runtime that failed its boot is parked and still answers ``docker exec`` with the
provenance of the image it was built from. The follow read that as the lane running
the tag and answered ``up_to_date`` for a lane that was crash-parked.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest
from deploy_agent.executor import load_dev_boot_guard
from deploy_agent.idle_converge import (
    EnumStabilityFollowVerdict,
    ModelDevTag,
    ModelStabilityFollowInputs,
    decide_stability_follow,
)

pytestmark = pytest.mark.unit

TAG = ModelDevTag(
    name="dev-20261115000000-abcdef123456",
    sha="a" * 40,
    cut_at=datetime(2026, 11, 15, tzinfo=UTC),
)
CONTAINER = "stability-main"


def _inputs(running_ref: str | None) -> ModelStabilityFollowInputs:
    return ModelStabilityFollowInputs(
        tags=(TAG,),
        dev_ran_tag_sha=TAG.sha,
        dev_receipt="success",
        stability_running_ref=running_ref,
        job_active=False,
    )


def _docker_says(stdout: str, returncode: int = 0) -> Any:
    return patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess([], returncode, stdout, ""),
    )


def _read_ref(container: str, repository: str) -> str | None:
    assert container == CONTAINER
    return TAG.sha


def test_an_unhealthy_runtime_reads_as_running_no_revision() -> None:
    guard = load_dev_boot_guard()
    with _docker_says("unhealthy\n"):
        assert guard.running_ref_if_healthy(CONTAINER, "omnimarket", _read_ref) == ""


def test_a_healthy_runtime_reads_its_provenance() -> None:
    guard = load_dev_boot_guard()
    with _docker_says("healthy\n"):
        assert guard.running_ref_if_healthy(CONTAINER, "omnimarket", _read_ref) == (
            TAG.sha
        )


def test_the_reader_the_agent_installs_answers_by_repository() -> None:
    reader = load_dev_boot_guard().healthy_ref_reader(lambda: CONTAINER, _read_ref)
    with _docker_says("unhealthy\n"):
        assert reader("omnimarket") == ""
    with _docker_says("healthy\n"):
        assert reader("omnimarket") == TAG.sha


def test_a_docker_failure_stays_unread() -> None:
    guard = load_dev_boot_guard()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 10)):
        assert guard.running_ref_if_healthy(CONTAINER, "omnimarket", _read_ref) is None


def test_follow_rebuilds_a_lane_whose_runtime_is_unhealthy() -> None:
    assert (
        decide_stability_follow(_inputs("")).verdict
        is EnumStabilityFollowVerdict.FOLLOW
    )


def test_control_follow_leaves_a_healthy_lane_on_the_tag_alone() -> None:
    assert (
        decide_stability_follow(_inputs(TAG.sha)).verdict
        is EnumStabilityFollowVerdict.UP_TO_DATE
    )


def _bad_stability_policy(lane: object) -> str:
    from deploy_agent.executor import EnumRuntimeLane

    if lane is EnumRuntimeLane.STABILITY_TEST:
        raise ValueError("stability fold policy unreadable")
    return "omninode-runtime"


def _new_agent(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from deploy_agent import agent as agent_mod

    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    return agent_mod.DeployAgent(skip_self_update=True)


def test_a_dev_agent_starts_with_a_bad_stability_fold_policy(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deploy_agent import agent as agent_mod
    from deploy_agent.executor import EnumRuntimeLane

    resolve = Mock(side_effect=_bad_stability_policy)
    monkeypatch.setattr(agent_mod, "_runtime_container_for_lane", resolve)
    agent = _new_agent(tmp_path, monkeypatch)
    # Built in __init__, resolved on the stability lane only when the follow reads it.
    # The dev lane is read once at start, to seed the agent's own deploy-source clone.
    assert [call.args[0] for call in resolve.call_args_list] == [EnumRuntimeLane.DEV]
    # The stability lane alone fails: its read is unanswered, nothing raises.
    assert agent._stability_read_running_ref("omnimarket") is None


def test_the_agent_s_stability_reader_reads_the_stability_runtime(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deploy_agent import agent as agent_mod
    from deploy_agent.executor import EnumRuntimeLane

    read_ref = Mock(return_value=TAG.sha)
    monkeypatch.setattr(agent_mod, "read_running_ref", read_ref)
    agent = _new_agent(tmp_path, monkeypatch)
    read_ref.reset_mock()  # the start read the dev runtime once to seed the clone
    container = agent_mod._runtime_container_for_lane(EnumRuntimeLane.STABILITY_TEST)
    with _docker_says("unhealthy\n"):
        assert agent._stability_read_running_ref("omnimarket") == ""
    read_ref.assert_not_called()
    with _docker_says("healthy\n"):
        assert agent._stability_read_running_ref("omnimarket") == TAG.sha
    read_ref.assert_called_once_with(container, "omnimarket")
