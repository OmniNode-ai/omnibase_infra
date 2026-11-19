# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every dev recreate is announced on the dev bus before it runs (OMN-25912).

Run with ``-k recreate_announce``. The topic is declared in the node contract of
``node_dev_roll_events_effect``; no module here or in the handler types it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
import yaml
from deploy_agent import agent as agent_mod
from deploy_agent.job_state import JobStore

from omnibase_infra.enums.enum_dev_recreate_reason import EnumDevRecreateReason
from omnibase_infra.handlers.handler_dev_recreate_window import (
    DevRecreateWindowGate,
    RecreateNotAnnouncedError,
)
from omnibase_infra.models.model_dev_recreate_announcement import (
    ModelDevRecreateAnnouncement,
)
from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_roll_events import (
    DevRollEventsPublisher,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
NODE = REPO_ROOT / "src" / "omnibase_infra" / "nodes" / "node_dev_roll_events_effect"
POLICY = ModelDevRecreatePolicy(
    window_seconds=3600, offset_seconds=0, start_grace_seconds=300
)
OPEN = datetime(2026, 11, 19, 6, 0, 20, tzinfo=UTC)
CLOSED = datetime(2026, 11, 19, 5, 12, 28, tzinfo=UTC)


class _Log:
    def __init__(self, accepts: bool = True) -> None:
        self.accepts = accepts
        self.events: list[str] = []
        self.sent: list[ModelDevRecreateAnnouncement] = []

    def announce(self, announcement: ModelDevRecreateAnnouncement) -> bool:
        self.events.append("announce")
        self.sent.append(announcement)
        return self.accepts

    def recreate(self) -> None:
        self.events.append("recreate")


def test_recreate_announce_precedes_the_recreate() -> None:
    log = _Log()
    gate = DevRecreateWindowGate(POLICY, log.announce)
    assert gate.run_recreate("agent/idle-converge", OPEN, log.recreate) is True
    assert log.events == ["announce", "recreate"]
    assert log.sent[0].immediate is True
    assert log.sent[0].reason is EnumDevRecreateReason.WINDOW_OPEN


def test_recreate_announce_a_recreate_with_no_announcement_is_refused() -> None:
    log = _Log(accepts=False)
    gate = DevRecreateWindowGate(POLICY, log.announce)
    with pytest.raises(RecreateNotAnnouncedError):
        gate.run_recreate("agent/idle-converge", OPEN, log.recreate)
    assert "recreate" not in log.events
    assert gate.last_recreate_at is None, (
        "a refused recreate is not recorded as started"
    )


def test_recreate_announce_a_publisher_that_raises_is_a_refusal() -> None:
    def broken(announcement: ModelDevRecreateAnnouncement) -> bool:
        raise OSError("bus down")

    ran: list[str] = []
    gate = DevRecreateWindowGate(POLICY, broken)
    with pytest.raises(RecreateNotAnnouncedError):
        gate.run_recreate("agent/idle-converge", OPEN, lambda: ran.append("recreate"))
    assert ran == []


def test_recreate_announce_a_hold_names_the_next_recreate_once() -> None:
    log = _Log()
    gate = DevRecreateWindowGate(POLICY, log.announce)
    for _ in range(3):
        assert gate.permit("agent/idle-converge", CLOSED) is False
    assert len(log.sent) == 1
    held = log.sent[0]
    assert held.immediate is False
    assert held.next_recreate_at == datetime(2026, 11, 19, 6, 0, tzinfo=UTC)
    assert held.reason is EnumDevRecreateReason.WINDOW_MISSED


def test_recreate_announce_agent_refuses_an_unannounced_idle_converge(
    tmp_path: Path,
) -> None:
    log = _Log(accepts=False)
    gate = DevRecreateWindowGate(POLICY, log.announce, clock=lambda: OPEN)
    agent = agent_mod.DeployAgent.__new__(agent_mod.DeployAgent)
    agent.__dict__.update(
        _load_gate=None,
        _load_gate_last_verdict=None,
        _recreate_gate=gate,
        job_store=JobStore(tmp_path),
    )
    assert agent._idle_converge_gate_open("agent/idle-converge") is False
    assert gate.last_recreate_at is None


def test_recreate_announce_topic_is_declared_in_the_node_contract() -> None:
    contract = yaml.safe_load((NODE / "contract.yaml").read_text(encoding="utf-8"))
    topic = contract["topics"]["dev_recreate_announced"]
    sent: list[tuple[str, str, dict[str, object]]] = []
    publisher = DevRollEventsPublisher(
        lambda t, k, p: sent.append((t, k, p)) or True,
        {key: str(value) for key, value in contract["topics"].items()},
    )
    announcement = ModelDevRecreateAnnouncement(
        event_id=UUID("0f0e0d0c-0b0a-4908-8706-050403020100"),
        runtime_lane="dev",
        requested_by="agent/idle-converge",
        announced_at=OPEN,
        next_recreate_at=OPEN,
        reason=EnumDevRecreateReason.WINDOW_OPEN,
        immediate=True,
    )
    assert publisher.recreate_announced(announcement) is True
    assert sent[0][0] == topic
    assert sent[0][2]["next_recreate_at"] and sent[0][2]["reason"] == "window_open"
    operations = [h["operation"] for h in contract["handler_routing"]["handlers"]]
    assert "dev_roll.announce_recreate" in operations


def test_recreate_announce_no_topic_string_is_typed_in_code() -> None:
    for path in (
        REPO_ROOT / "src/omnibase_infra/handlers/handler_dev_recreate_window.py",
        NODE / "dev_roll_events.py",
        NODE / "runtime_dev_roll_events.py",
    ):
        assert "onex.evt." not in path.read_text(encoding="utf-8"), path
