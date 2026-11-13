# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev deploy that fails in preflight publishes a manifest and an incident (OMN-24048)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import kafka_config as kafka_config_mod
from deploy_agent import publisher as publisher_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.executor import DeployExecutor, load_dev_boot_guard
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

TOPIC_MANIFEST = "onex.evt.omnimarket.dev-roll-manifest.v1"
TOPIC_INCIDENT = "onex.evt.omnimarket.dev-roll-incident.v1"


def _agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cmd: ModelRebuildRequested
) -> tuple[DeployAgent, Mock]:
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod.socket, "gethostname", lambda: "preflight-test")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = JobStore(tmp_path)
    agent.job_store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    executor = Mock(spec=DeployExecutor)
    executor.git_pull.return_value = "a" * 40
    executor.rebuild_scope.return_value = ["omninode-runtime"]
    executor.verify.return_value = []
    executor.deploy_and_verify.return_value = []
    executor.run_lane_mirror.return_value = None
    executor.resolve_stability_ready_digest.return_value = cmd.image_digest
    for name in (
        "container_residue",
        "recreate_supervision",
        "verify_recreate",
        "oneshot_reruns",
        "deps_convergence",
        "compose_invocations",
        "health_checks",
    ):
        setattr(executor, name, [])
    executor.cloud_migration_failure = ""
    executor.sibling_source_refs = {}
    agent.executor = executor
    monkeypatch.setattr(agent, "_resolve_lab_overlay_sha", lambda *args, **kwargs: None)
    return agent, executor


def _dev_cmd() -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="gha/omnimarket/pr-5942",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )


def _wire_sender(
    monkeypatch: pytest.MonkeyPatch, *, sender_raises: bool = False
) -> list[tuple[str, str, dict[str, Any]]]:
    """Replace the agent's own producer (the layer under agent_send), not the events module."""
    sent: list[tuple[str, str, dict[str, Any]]] = []

    def produce(topic: str, key: str, payload: dict[str, Any], config: object) -> bool:
        if sender_raises:
            raise OSError("broker down")
        sent.append((topic, key, payload))
        return True

    monkeypatch.setattr(publisher_mod, "_ensure_topic", lambda topic, config: None)
    monkeypatch.setattr(publisher_mod, "_produce", produce)
    return sent


def test_preflight_failure_publishes_manifest_then_incident_and_job_stays_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = _dev_cmd()
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    executor.preflight.side_effect = RuntimeError("Git remote unreachable")
    sent = _wire_sender(monkeypatch)

    agent._run_deploy(cmd)

    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert job.errors == ["Git remote unreachable"]
    assert len(sent) == 2
    assert sent[0][0] == TOPIC_MANIFEST
    assert sent[1][0] == TOPIC_INCIDENT
    manifest = sent[0][2]
    assert manifest["job_id"] == str(cmd.correlation_id)
    assert manifest["issuer"] == "gha/omnimarket/pr-5942"
    incident = sent[1][2]
    assert incident["issuer"] == "gha/omnimarket/pr-5942"
    assert incident["failed_phase"] == "preflight"
    assert incident["job_id"] == str(cmd.correlation_id)
    assert "Git remote unreachable" in incident["error"]
    assert incident["recovery_verified"] is False
    assert set(incident) == {
        "event_id",
        "job_id",
        "issuer",
        "commit",
        "error",
        "project",
        "at",
        "failed_phase",
        "restored_revision",
        "recovery_verified",
        "recovery_skipped_healthy",
        "recovery_error",
        "runtime_kept_revision",
        "health_probes",
        "log_captures",
    }


def test_preflight_failure_sender_error_still_fails_job_without_propagating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = _dev_cmd()
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    executor.preflight.side_effect = RuntimeError("Git remote unreachable")
    _wire_sender(monkeypatch, sender_raises=True)

    agent._run_deploy(cmd)

    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert job.errors == ["Git remote unreachable"]


def test_stability_test_preflight_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="gha/omnimarket/pr-5942",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.STABILITY_TEST,
    )
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    executor.preflight.side_effect = RuntimeError("Git remote unreachable")
    sent = _wire_sender(monkeypatch)

    agent._run_deploy(cmd)

    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert sent == []


def test_unbuildable_events_never_replace_the_preflight_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = _dev_cmd()
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    executor.preflight.side_effect = RuntimeError("Git remote unreachable")
    sent = _wire_sender(monkeypatch)
    monkeypatch.setattr(
        kafka_config_mod,
        "load_deploy_agent_kafka_config_from_env",
        Mock(side_effect=ValueError("no broker settings")),
    )

    agent._run_deploy(cmd)

    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert job.errors == ["Git remote unreachable"]
    assert sent == []


def test_run_preflight_announces_a_dev_failure_and_raises_the_same_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _wire_sender(monkeypatch)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    executor = Mock()
    executor.preflight.side_effect = RuntimeError("Docker unavailable")

    with pytest.raises(RuntimeError, match="Docker unavailable"):
        load_dev_boot_guard().run_preflight(executor, _dev_cmd(), Mock())

    assert [topic for topic, _key, _payload in sent] == [TOPIC_MANIFEST, TOPIC_INCIDENT]
    assert sent[1][2]["failed_phase"] == "preflight"
    assert sent[1][2]["issuer"] == "gha/omnimarket/pr-5942"


def test_run_preflight_that_passes_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _wire_sender(monkeypatch)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    executor = Mock()
    callback = Mock()

    load_dev_boot_guard().run_preflight(executor, _dev_cmd(), callback)

    executor.preflight.assert_called_once_with(on_phase_update=callback)
    assert sent == []
