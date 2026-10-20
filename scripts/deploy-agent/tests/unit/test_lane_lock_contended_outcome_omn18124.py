# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18124 AC4: lane lock contention has a typed, durable job outcome."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    EnumJobNotStarted,
    EnumRuntimeLane,
    ModelRebuildCompleted,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.host_slot import ENV_HOST_SLOT_DIR, HostSlot
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.lane_lock_client import LaneLockContendedError

pytestmark = pytest.mark.unit

SHA = "a" * 40


class _FakeExecutor:
    def __init__(self, slot: HostSlot | None = None) -> None:
        self.calls: list[str] = []
        self.slot = slot
        self.holder_during_build: str | None = None
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.oneshot_reruns: list[object] = []
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []

    def preflight(self, **kwargs: object) -> None:
        self.calls.append("preflight")

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        self.calls.append("git_pull")
        return SHA

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None:
        self.calls.append("compose_gen")

    def seed_infisical(self, **kwargs: object) -> None:
        self.calls.append("seed_infisical")

    def validate_llm_endpoint_env_contract(self) -> None:
        self.calls.append("validate_llm_endpoint_env_contract")

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        self.calls.append("rebuild_scope")
        if self.slot is not None:
            holder = self.slot.holder()
            self.holder_during_build = holder.owner if holder else None
        return ["omninode-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        self.calls.append("verify")
        return []

    def run_lane_mirror(self, **kwargs: object) -> None:
        return None


def _agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, slot_dir: Path | None
) -> tuple[DeployAgent, JobStore, ModelRebuildRequested]:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    if slot_dir is None:
        monkeypatch.delenv(ENV_HOST_SLOT_DIR, raising=False)
    else:
        monkeypatch.setenv(ENV_HOST_SLOT_DIR, str(slot_dir))
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    store = JobStore(tmp_path / "jobs")
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    return agent, store, cmd


def _capture_payloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    def publish(payload: dict[str, Any], config: object) -> bool:
        payloads.append(payload)
        return True

    monkeypatch.setattr(agent_mod, "publish_result", publish)
    return payloads


@contextmanager
def _uncontended_lane_lock(*args: object, **kwargs: object) -> Iterator[None]:
    yield


def test_lane_lock_contended_job_and_terminal_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, store, cmd = _agent(tmp_path, monkeypatch, slot_dir=None)
    fake = _FakeExecutor()
    monkeypatch.setattr(agent, "executor", fake)
    payloads = _capture_payloads(monkeypatch)
    holder = "lane dev held by refresh_dev_lane.sh pid 4242"

    class _Contended:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> None:
            raise LaneLockContendedError(holder)

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(agent_mod, "lane_lock", _Contended)

    agent._run_deploy(cmd)

    assert fake.calls == []
    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert job.not_started == EnumJobNotStarted.LANE_LOCK_CONTENDED
    assert holder in job.errors
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["not_started"] == "lane_lock_contended"
    assert payload["status"] == "failed"
    # status is computed on output and refused as an input by extra="forbid".
    completed = ModelRebuildCompleted.model_validate(
        {key: value for key, value in payload.items() if key != "status"}
    )
    assert completed.not_started == EnumJobNotStarted.LANE_LOCK_CONTENDED
    assert completed.model_dump(mode="json") == payload


def test_lane_lock_contended_build_failure_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BuildFailingExecutor(_FakeExecutor):
        def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
            self.calls.append("rebuild_scope")
            raise RuntimeError("compose build failed")

    agent, store, cmd = _agent(tmp_path, monkeypatch, slot_dir=None)
    fake = BuildFailingExecutor()
    monkeypatch.setattr(agent, "executor", fake)
    payloads = _capture_payloads(monkeypatch)
    monkeypatch.setattr(agent_mod, "lane_lock", _uncontended_lane_lock)

    agent._run_deploy(cmd)

    assert "rebuild_scope" in fake.calls
    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert job.not_started is None
    assert "compose build failed" in job.errors
    assert len(payloads) == 1
    assert payloads[0]["not_started"] is None
    assert payloads[0]["status"] == "failed"


def test_lane_lock_contended_older_job_record_loads_without_reason(
    tmp_path: Path,
) -> None:
    cid = uuid4()
    record = JobState(correlation_id=cid, command={}, status="failed").model_dump(
        mode="json", exclude={"not_started"}
    )
    assert "not_started" not in record
    (tmp_path / f"{cid}.json").write_text(json.dumps(record))

    job = JobStore(tmp_path).load(cid)

    assert job is not None
    assert job.status == "failed"
    assert job.not_started is None


def test_lane_lock_contended_complete_persists_reason(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    cid = uuid4()
    store.accept(cid, {})

    completed = store.complete(
        cid,
        status="failed",
        errors=["lane dev held by refresh_dev_lane.sh pid 4242"],
        not_started=EnumJobNotStarted.LANE_LOCK_CONTENDED,
    )

    assert completed.not_started == EnumJobNotStarted.LANE_LOCK_CONTENDED
    job = JobStore(tmp_path).load(cid)
    assert job is not None
    assert job.status == "failed"
    assert job.not_started == EnumJobNotStarted.LANE_LOCK_CONTENDED
    assert job.errors == completed.errors
