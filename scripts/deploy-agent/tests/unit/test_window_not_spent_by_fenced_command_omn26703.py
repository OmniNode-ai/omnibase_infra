# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A command a fence refuses never spends the recreate window (OMN-26703).

On 2026-11-27 the 14:00Z and 15:00Z windows were each taken by a request the agent then
rejected (a contained, already-built ref, and a duplicate of it). The consumer asked the
window before dedup and the lineage fence, the first admitted ask spends the window, so
the real build requests behind it read ``window_spent`` and waited an hour each.

Run with ``-k window_not_spent``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested

pytestmark = pytest.mark.unit


def _lane_build() -> ModelRebuildRequested:
    return ModelRebuildRequested.model_validate(
        {
            "correlation_id": str(uuid4()),
            "requested_by": "agent/overdue-merge",
            "scope": "full",
            "runtime_lane": "dev",
            "build_source": "workspace",
            "git_ref": "a" * 40,
            "rollback": {"actor": "agent/overdue-merge", "reason": "overdue merge"},
        }
    )


def _message(cmd: ModelRebuildRequested, offset: int) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json") | {"_signature": "a" * 64}
    return SimpleNamespace(
        value=payload, topic="t", partition=0, offset=offset, key=None, timestamp=None
    )


class _OneWindowGate:
    """The real gate's rule: the first admitted ask spends the window."""

    def __init__(self) -> None:
        self.asked = 0
        self.spent = False
        self.last_decision = SimpleNamespace(
            reason=SimpleNamespace(value="window_spent"),
            next_recreate_at=datetime.now(UTC),
        )

    def allow(self, requested_by: str, observed_last: object = None) -> bool:
        self.asked += 1
        if self.spent:
            return False
        self.spent = True
        return True


def _consumer(gate: _OneWindowGate) -> DeployConsumer:
    roll = agent_mod.load_dev_boot_guard(agent_mod.DEV_ROLL_EVENTS)
    window = roll.lane_build_gate(SimpleNamespace(_recreate_gate=gate, _load_gate=None))
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


def _process(consumer: DeployConsumer, cmd: ModelRebuildRequested, offset: int) -> Any:
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        return consumer._process_message(_message(cmd, offset))


def test_window_not_spent_by_a_duplicate_then_the_real_build_runs() -> None:
    gate = _OneWindowGate()
    consumer = _consumer(gate)
    duplicate, real = _lane_build(), _lane_build()
    consumer.job_store.is_duplicate.side_effect = lambda cid: (
        cid == duplicate.correlation_id
    )

    assert _process(consumer, duplicate, 1) == (None, "duplicate")
    assert gate.asked == 0, "a duplicate never asks the window"
    assert _process(consumer, real, 2) == (real, None)
    consumer.consumer.pause.assert_not_called()


def test_window_not_spent_by_a_superseded_command_then_the_real_build_runs() -> None:
    gate = _OneWindowGate()
    consumer = _consumer(gate)
    contained, real = _lane_build(), _lane_build()
    supersedes = SimpleNamespace(
        verdict=SimpleNamespace(
            refuses=False, supersedes=True, permits_coalescing=False
        ),
        journal_line=lambda: "contained",
    )

    def lineage(cmd: ModelRebuildRequested, msg: object) -> Any:
        return supersedes if cmd.correlation_id == contained.correlation_id else None

    with (
        patch.object(consumer, "_lineage_decision", side_effect=lineage),
        patch.object(consumer, "_command_payload", return_value={}),
        patch("deploy_agent.consumer.verify_command", return_value=True),
    ):
        first = consumer._process_message(_message(contained, 1))
        assert gate.asked == 0, "a superseded command never asks the window"
        assert _process(consumer, real, 2) == (real, None)
    assert first[0] is None
    consumer.consumer.pause.assert_not_called()
