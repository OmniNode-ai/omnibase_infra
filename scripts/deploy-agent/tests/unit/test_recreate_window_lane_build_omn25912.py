# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane's dev build request asks the recreate window; a real rollback goes at once (OMN-25912).

A lane asks the dev lane for a build of its branch with a rollback declaration, because the
declaration is the one way past the lineage fence. The window gate covered the agent's idle
paths only, so lane builds at 20:07:31Z and 21:34:00Z on h201 recreated the runtime inside an
hour that already held a recreate. The consumer now asks the window for such a request and
defers it, uncommitted and unpublished, to the next window; a rollback to a pinned dev tag
or image still runs at once.

Run with ``-k recreate_window``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildRequested,
)

pytestmark = pytest.mark.unit

SHA = "a" * 40
REASON = "window_spent"


def _command(**extra: object) -> ModelRebuildRequested:
    return ModelRebuildRequested.model_validate(
        {
            "correlation_id": str(uuid4()),
            "requested_by": "lane/ls25912c",
            "scope": "full",
            "runtime_lane": "dev",
            "build_source": "workspace",
            "git_ref": SHA,
            **extra,
        }
    )


def _lane_build() -> ModelRebuildRequested:
    return _command(rollback={"actor": "lane/ls25912c", "reason": "build my branch"})


def _pinned_tag_rollback() -> ModelRebuildRequested:
    return _command(
        git_ref="dev-20261125010101-abcdef1234567",
        rollback={"actor": "operator", "reason": "restore the known good tag"},
    )


def _pinned_image_rollback() -> ModelRebuildRequested:
    return _command(
        image_ref="ghcr.io/omninode/runtime:good",
        image_digest="sha256:" + "b" * 64,
        rollback={"actor": "operator", "reason": "restore the known good image"},
    )


def _message(cmd: ModelRebuildRequested) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json") | {"_signature": "a" * 64}
    return SimpleNamespace(
        value=payload, topic="t", partition=0, offset=100, key=None, timestamp=None
    )


class _Gate:
    """The agent's recreate gate: shut until ``until``, naming why, as the real gate does."""

    def __init__(self, until: datetime) -> None:
        self.until = until
        self.asked: list[str] = []
        self.last_decision: object | None = None

    def allow(self, requested_by: str, observed_last: object = None) -> bool:
        self.asked.append(requested_by)
        if datetime.now(UTC) >= self.until:
            return True
        self.last_decision = SimpleNamespace(
            reason=SimpleNamespace(value="window_spent"), next_recreate_at=self.until
        )
        return False


def _window(gate: _Gate, inner: Any = None) -> Any:
    roll = agent_mod.load_dev_boot_guard(agent_mod.DEV_ROLL_EVENTS)
    agent = SimpleNamespace(_recreate_gate=gate, _load_gate=inner)
    return roll.lane_build_gate(agent)


def _consumer(window: Any) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.consumer.poll.return_value = {}
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.notices: list[Any] = []
    consumer.on_rejected = consumer.notices.append
    consumer.load_gate = window
    return consumer


def _process(consumer: DeployConsumer, cmd: ModelRebuildRequested) -> tuple:
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        return consumer._process_message(_message(cmd))


def _in_an_hour() -> datetime:
    return datetime.now(UTC) + timedelta(hours=1)


def test_recreate_window_defers_a_lane_build_inside_the_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = _Gate(_in_an_hour())
    consumer = _consumer(_window(gate))

    with caplog.at_level("INFO"):
        assert _process(consumer, _lane_build()) == (None, None)

    assert gate.asked == ["lane/ls25912c"]
    consumer.job_store.accept.assert_not_called()
    consumer.consumer.commit.assert_not_called()
    assert consumer.notices == []
    (tp, offset), _ = consumer.consumer.seek.call_args
    assert (tp.topic, tp.partition, offset) == ("t", 0, 100)
    consumer.consumer.pause.assert_called_once_with(tp)
    assert "deferred to the recreate window" in caplog.text
    assert REASON in caplog.text, "the deferral names its reason"


def test_recreate_window_admits_a_lane_build_in_the_open_window() -> None:
    cmd = _lane_build()
    consumer = _consumer(_window(_Gate(datetime.now(UTC) - timedelta(seconds=1))))

    assert _process(consumer, cmd) == (cmd, None)
    consumer.consumer.pause.assert_not_called()


@pytest.mark.parametrize("make", [_pinned_tag_rollback, _pinned_image_rollback])
def test_recreate_window_lets_a_real_rollback_go_at_once(make: Any) -> None:
    gate = _Gate(_in_an_hour())
    consumer = _consumer(_window(gate))
    cmd = make()

    assert _process(consumer, cmd) == (cmd, None)
    assert gate.asked == [], "a rollback to a pinned good build never asks the window"
    consumer.consumer.pause.assert_not_called()


def test_recreate_window_does_not_hold_a_command_with_no_rollback() -> None:
    gate = _Gate(_in_an_hour())
    consumer = _consumer(_window(gate))
    cmd = _command()

    assert _process(consumer, cmd) == (cmd, None)
    assert gate.asked == []


def test_recreate_window_does_not_hold_a_non_dev_lane() -> None:
    gate = _Gate(_in_an_hour())
    consumer = _consumer(_window(gate))
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.STABILITY_TEST})
    cmd = _command(
        runtime_lane=EnumRuntimeLane.STABILITY_TEST.value,
        rollback={"actor": "lane/x", "reason": "build"},
    )

    _process(consumer, cmd)
    assert gate.asked == []


def test_recreate_window_resumes_the_partition_when_the_window_opens() -> None:
    gate = _Gate(_in_an_hour())
    consumer = _consumer(_window(gate))
    _process(consumer, _lane_build())
    paused = consumer.load_gate_paused

    consumer.poll_and_accept()
    consumer.consumer.resume.assert_not_called()

    consumer.load_gate._until = datetime.now(UTC) - timedelta(
        seconds=1
    )  # the next window starts
    consumer.poll_and_accept()
    consumer.consumer.resume.assert_called_once_with(paused)
    assert consumer.load_gate_paused is None


def test_recreate_window_consumer_without_a_window_is_unchanged() -> None:
    cmd = _lane_build()
    assert _process(
        _consumer(_window(_Gate(datetime.now(UTC) - timedelta(seconds=1)))), cmd
    ) == (cmd, None)
