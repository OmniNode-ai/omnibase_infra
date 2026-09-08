# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16939: the deploy agent's lane fence is required and fail-closed.

RED before the fence existed: a DeployConsumer built with no
DEPLOY_AGENT_ALLOWED_LANES accepted a stability-test command on the dev bus.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.lane_policy import (
    ENV_ALLOWED_LANES,
    LaneNotAllowedError,
    assert_lane_allowed,
    load_allowed_lanes_from_env,
    parse_allowed_lanes,
)


def test_parse_single_lane() -> None:
    assert parse_allowed_lanes("dev") == frozenset({EnumRuntimeLane.DEV})


def test_parse_multiple_lanes_ignores_whitespace() -> None:
    assert parse_allowed_lanes(" dev , stability-test ") == frozenset(
        {EnumRuntimeLane.DEV, EnumRuntimeLane.STABILITY_TEST}
    )


@pytest.mark.parametrize("raw", ["", "   ", ",", " , "])
def test_parse_rejects_empty_list(raw: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_allowed_lanes(raw)


def test_parse_rejects_unknown_lane_rather_than_dropping_it() -> None:
    """A typo must abort, not silently narrow the fence."""
    with pytest.raises(ValueError, match="unknown runtime lane"):
        parse_allowed_lanes("dev,stabilty-test")


def test_env_loader_is_required_with_no_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_ALLOWED_LANES, raising=False)
    with pytest.raises(RuntimeError, match="is required for deploy-agent"):
        load_allowed_lanes_from_env()


def test_env_loader_reads_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ALLOWED_LANES, "dev")
    assert load_allowed_lanes_from_env() == frozenset({EnumRuntimeLane.DEV})


def test_assert_allows_a_fenced_lane() -> None:
    assert_lane_allowed(EnumRuntimeLane.DEV, frozenset({EnumRuntimeLane.DEV}))


@pytest.mark.parametrize("lane", [EnumRuntimeLane.STABILITY_TEST, EnumRuntimeLane.PROD])
def test_assert_refuses_every_lane_outside_the_fence(lane: EnumRuntimeLane) -> None:
    """The dev control bus carries stability-test commands too — this is the
    exact case the fence exists for."""
    with pytest.raises(LaneNotAllowedError, match=lane.value):
        assert_lane_allowed(lane, frozenset({EnumRuntimeLane.DEV}))


def _consumer_with_fence() -> object:
    """A DeployConsumer with no broker attached, fenced to the dev lane.

    Same construction shape as tests/unit/test_consumer_signed_payload.py.
    """
    from unittest.mock import Mock

    from deploy_agent.consumer import DeployConsumer

    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    # OMN-16442: the pre-accept self-update boundary. These tests are not
    # testing self-update, so the hook is an explicit no-op rather than absent
    # -- an absent attribute would be swallowed by the boundary's own
    # error rail and read as a pass.
    consumer.self_update_hook = lambda rewind: None
    return consumer


def _payload(lane: str) -> dict[str, object]:
    return {
        "correlation_id": str(uuid.uuid4()),
        "requested_by": "gha/omnibase_infra/pr-0000",
        "scope": "full",
        "runtime_lane": lane,
        "build_source": "workspace",
        "services": [],
        "git_ref": "origin/dev",
        "_signature": "a" * 64,
    }


@pytest.mark.unit
def test_consumer_rejects_an_off_lane_command_and_commits_the_offset() -> None:
    """The dev control bus carries stability-test commands by design.

    RED before the fence: this returned an accepted command and the agent
    would have rebuilt the stability-test lane.
    """
    from unittest.mock import patch

    consumer = _consumer_with_fence()
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        cmd, reason = consumer._process_message(
            SimpleNamespace(value=_payload("stability-test"))
        )

    assert cmd is None
    assert reason == "lane_not_allowed"
    consumer.job_store.accept.assert_not_called()
    consumer.job_store.has_active_job.assert_not_called()
    # the offset must still commit: the command is not for this agent, and
    # re-reading it forever would stall every later command behind it.
    consumer.consumer.commit.assert_called_once()


@pytest.mark.unit
def test_consumer_accepts_an_in_lane_command() -> None:
    """Positive control for the test above: the same path accepts lane=dev."""
    from unittest.mock import patch

    consumer = _consumer_with_fence()
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        cmd, reason = consumer._process_message(SimpleNamespace(value=_payload("dev")))

    assert reason is None
    assert cmd is not None
    assert cmd.runtime_lane is EnumRuntimeLane.DEV
    consumer.job_store.accept.assert_called_once()
