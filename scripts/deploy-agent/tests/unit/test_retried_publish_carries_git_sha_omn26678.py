# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A replayed rebuild-completed event carries the sha the job resolved (OMN-26678).

The publish retry loop rebuilt the completion payload with an empty ``git_sha``, because
the sha lived only on the agent's per-job field, which the next job resets. The delegation
checks node refuses an empty ``git_sha`` (``ModelRebuildCompletedObserved``), so every
build whose first publish failed got no check result. The sha is now durable on the job
record and the retry reads it from there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus

pytestmark = pytest.mark.unit

SHA = "b1dbc24e33e1c6d9f7a2c3b4d5e6f708192a3b4c"


def _agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeployAgent:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    agent = DeployAgent(skip_self_update=True)
    agent._allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    return agent


def _owed_job(agent: DeployAgent, *, resolved_sha: str | None) -> Any:
    cid = uuid4()
    agent.job_store.accept(
        cid,
        {
            "correlation_id": str(cid),
            "git_ref": "b1dbc24e33",
            "runtime_lane": "dev",
            "scope": "runtime",
        },
    )
    if resolved_sha is not None:
        agent.job_store.record_git_sha(cid, resolved_sha)
    agent.job_store.update_phase(cid, Phase.GIT, PhaseStatus.SUCCESS)
    agent.job_store.complete(cid, status="success")
    job = agent.job_store.load(cid)
    job.result_publish_pending = True
    agent.job_store._save(job)
    return cid


def test_retried_publish_carries_the_resolved_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(
        agent_mod,
        "publish_result",
        lambda payload, config: sent.append(payload) or True,
    )
    _owed_job(agent, resolved_sha=SHA)
    # The next job resets the agent's per-job field before the retry tick runs.
    agent._current_git_sha = ""

    agent._retry_pending_publishes()

    assert [p["git_sha"] for p in sent] == [SHA]


def test_retried_publish_of_an_unresolved_job_stays_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: a job that never resolved a sha reports none, beside one that did."""
    agent = _agent(tmp_path, monkeypatch)
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(
        agent_mod,
        "publish_result",
        lambda payload, config: sent.append(payload) or True,
    )
    resolved = _owed_job(agent, resolved_sha=SHA)
    unresolved = _owed_job(agent, resolved_sha=None)
    agent._current_git_sha = SHA

    agent._retry_pending_publishes()

    by_job = {str(p["correlation_id"]): p["git_sha"] for p in sent}
    assert by_job == {str(resolved): SHA, str(unresolved): ""}
