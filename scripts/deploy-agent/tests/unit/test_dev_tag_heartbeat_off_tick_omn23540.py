# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23540: the dev-tag heartbeat publish must not block the idle tick.

The synchronous Kafka send+flush used to run on the tick; where a broker
accepts the TCP connection and never answers it blocks 60 seconds. The send
now runs on its own daemon thread, so a blocked publish cannot hold the tick.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.idle_converge import DEV_TAG_HEARTBEAT_INTERVAL

from .test_idle_converge_omn19509 import _agent
from .test_lab_follows_dev_tag_omn19998 import NEW_CUT, TAGS, _join

pytestmark = pytest.mark.unit


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    return _agent(tmp_path, monkeypatch, omnimarket_to_202=False)


def test_a_blocked_publish_does_not_block_the_idle_tick(
    agent: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    entered = threading.Event()
    calls = 0

    def blocked_publish(payload: dict[str, Any], config: Any) -> bool:
        nonlocal calls
        calls += 1
        entered.set()
        release.wait(30.0)
        return True

    monkeypatch.setattr(agent_mod, "publish_dev_tag_heartbeat", blocked_publish)
    agent._idle_converge_now = lambda: NEW_CUT + timedelta(hours=2, minutes=5)
    agent._idle_read_running_ref = lambda repo: "2" * 40
    agent._idle_read_dev_tags = lambda repo: TAGS
    agent._idle_running_contains = lambda repo, ancestor, running: False

    worker = threading.Thread(
        target=agent._maybe_idle_converge, daemon=True, name="idle-tick"
    )
    worker.start()
    worker.join(timeout=5.0)
    try:
        assert not worker.is_alive()
        assert entered.is_set()
    finally:
        release.set()
        _join(agent)


def test_a_second_tick_inside_the_interval_starts_no_second_publish(
    agent: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    calls = 0

    def blocked_publish(payload: dict[str, Any], config: Any) -> bool:
        nonlocal calls
        calls += 1
        release.wait(30.0)
        return True

    monkeypatch.setattr(agent_mod, "publish_dev_tag_heartbeat", blocked_publish)
    now = NEW_CUT + timedelta(hours=2, minutes=5)
    agent._idle_converge_now = lambda: now
    agent._idle_read_running_ref = lambda repo: "2" * 40
    agent._idle_read_dev_tags = lambda repo: TAGS
    agent._idle_running_contains = lambda repo, ancestor, running: False

    try:
        agent._maybe_idle_converge()
        agent._idle_converge_last_check = None
        agent._idle_converge_now = lambda: now + timedelta(minutes=1)
        agent._maybe_idle_converge()  # the first publish is still blocked
        assert calls <= 1
    finally:
        release.set()
        _join(agent)
    assert calls == 1


def test_a_published_heartbeat_is_recorded_as_published(
    agent: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = NEW_CUT + timedelta(hours=2, minutes=5)
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(
        agent_mod,
        "publish_dev_tag_heartbeat",
        lambda payload, config: sent.append(payload) or True,
    )
    agent._idle_converge_now = lambda: now
    agent._idle_read_running_ref = lambda repo: "2" * 40
    agent._idle_read_dev_tags = lambda repo: TAGS
    agent._idle_running_contains = lambda repo, ancestor, running: False

    agent._maybe_idle_converge()
    _join(agent)

    assert len(sent) == 1
    assert agent._dev_tag_heartbeat_published_at["omnimarket"] == now


def test_a_failed_publish_does_not_stop_the_tick_and_the_next_interval_sends_again(
    agent: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = NEW_CUT + timedelta(hours=2, minutes=5)
    calls = 0

    def refused_then_sent(payload: dict[str, Any], config: Any) -> bool:
        nonlocal calls
        calls += 1
        return calls > 1  # the broker refuses the first send, as the publisher reports

    monkeypatch.setattr(agent_mod, "publish_dev_tag_heartbeat", refused_then_sent)
    agent._idle_converge_now = lambda: now
    agent._idle_read_running_ref = lambda repo: "2" * 40
    agent._idle_read_dev_tags = lambda repo: TAGS
    agent._idle_running_contains = lambda repo, ancestor, running: False

    agent._maybe_idle_converge()
    _join(agent)
    assert calls == 1  # the tick finished although the send was refused

    agent._idle_converge_last_check = None
    agent._idle_converge_now = lambda: now + DEV_TAG_HEARTBEAT_INTERVAL
    agent._maybe_idle_converge()
    _join(agent)
    assert calls == 2
