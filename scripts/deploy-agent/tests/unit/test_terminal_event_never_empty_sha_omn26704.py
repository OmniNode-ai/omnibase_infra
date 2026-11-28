# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26704 -- the rebuild-completed event never carries an empty git_sha.

``_run_deploy`` clears ``_current_git_sha`` at the start of every job and only
``git_pull`` sets it, so a job that ends before git_pull used to publish a
terminal event whose ``git_sha`` was empty; the consumer rejects that.

Exemplar: ``test_lab_overlay_build_order_omn18545`` (same fake executor and
agent harness, driving the real ``_run_deploy``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import publisher as publisher_mod

from .test_lab_overlay_build_order_omn18545 import (
    SHA,
    _dev_cmd,
    _FakeExecutor,
    _make_agent,
)

pytestmark = pytest.mark.unit


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    published: list[dict[str, Any]] = []

    def _produce(topic: str, key: str, payload: dict[str, Any], config: object) -> bool:
        # Other agent events share the wire; only the completed topic is judged.
        if topic == publisher_mod.TOPIC_REBUILD_COMPLETED:
            published.append(payload)
        return True

    # The wire call, below publish_result: the guard under test sits above it.
    monkeypatch.setattr(publisher_mod, "_produce", _produce)
    # _make_agent stubs publish_result out; put the real one back.
    monkeypatch.setattr(agent_mod, "publish_result", publisher_mod.publish_result)
    return published


def test_job_failing_before_git_pull_publishes_no_empty_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = _dev_cmd()
    fake = _FakeExecutor(preflight_error="preflight refused before git_pull")
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake)
    published = _capture(monkeypatch)

    agent._run_deploy(cmd)

    assert "git_pull" not in fake.calls, "control: the job must end before git_pull"
    assert all(p["git_sha"] for p in published), published
    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert not job.result_publish_pending, "nothing owed to the retry loop"


def test_normal_job_publishes_the_pulled_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal job publishes the pulled sha, and only that, on a shared agent.

    A job that ends before git_pull runs first on the same agent, so the
    published list is exactly what the consumer would see: the pulled sha, and
    no empty-sha event beside it.
    """
    early = _dev_cmd()
    agent, store = _make_agent(
        tmp_path, monkeypatch, early, _FakeExecutor(preflight_error="refused")
    )
    published = _capture(monkeypatch)
    agent._run_deploy(early)

    cmd = _dev_cmd()
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    fake = _FakeExecutor()
    monkeypatch.setattr(agent, "executor", fake)
    # The settle step after a successful roll would build an image in a real
    # subprocess (about 140 s, over the verifier's 30 s per-check ceiling) and
    # deliver the onex-api pin; neither is the unit under test, so name them off.
    monkeypatch.setattr(agent, "_build_onex_api_image", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_deliver_onex_api_pin", lambda *a, **k: None)
    agent._run_deploy(cmd)
    agent.drain_settle(timeout=10)

    assert "git_pull" in fake.calls
    assert [p["git_sha"] for p in published] == [SHA]
