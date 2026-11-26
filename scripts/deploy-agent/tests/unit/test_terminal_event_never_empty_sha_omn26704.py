# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26704: a rebuild-completed event never carries an empty git_sha.

``_run_deploy`` resets ``_current_git_sha`` at the start of every job and only
``git_pull`` sets it, so a job that ends before ``git_pull`` used to publish an
event with ``git_sha == ""`` (which a consumer rejects: min_length=1). The real
``publish_result`` is exercised here; only the Kafka producer beneath it is
replaced, so the assertions read what would reach the broker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import publisher

from .test_lane_lock_contended_outcome_omn18124 import (
    SHA,
    _agent,
    _FakeExecutor,
    _uncontended_lane_lock,
)

pytestmark = pytest.mark.unit


class _SucceedingExecutor(_FakeExecutor):
    cloud_migration_failure = None


class _PrePullFailingExecutor(_FakeExecutor):
    def preflight(self, **kwargs: object) -> None:
        self.calls.append("preflight")
        raise RuntimeError("preflight failed before git_pull")

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        raise AssertionError("git_pull must not be reached")


def _broker(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what the real publish_result hands to the producer."""
    sent: list[dict[str, Any]] = []

    def produce(topic: str, key: str, payload: dict[str, Any], config: object) -> bool:
        if topic == publisher.TOPIC_REBUILD_COMPLETED:
            sent.append(payload)
        return True

    monkeypatch.setattr(publisher, "_produce", produce)
    monkeypatch.setattr(agent_mod, "publish_result", publisher.publish_result)
    monkeypatch.setattr(agent_mod, "lane_lock", _uncontended_lane_lock)
    return sent


def test_job_failing_before_git_pull_publishes_no_empty_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, store, cmd = _agent(tmp_path, monkeypatch, slot_dir=None)
    monkeypatch.setattr(agent, "executor", _PrePullFailingExecutor())
    sent = _broker(monkeypatch)

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert [p for p in sent if not p["git_sha"]] == []
    assert not job.result_publish_pending


def test_normal_job_publishes_the_pulled_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _store, cmd = _agent(tmp_path, monkeypatch, slot_dir=None)
    monkeypatch.setattr(agent, "executor", _SucceedingExecutor())
    sent = _broker(monkeypatch)

    agent._run_deploy(cmd)

    assert len(sent) == 1
    assert sent[0]["git_sha"] == SHA
    assert sent[0]["status"] == "success"
