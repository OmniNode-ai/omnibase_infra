# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23521 -- the onex-api pin is delivered with the cluster apply deleted.

The dev roll's settle used to run the cluster apply and, as one step of it,
build the onex-api image the pin delivery advances to. The apply is retired
(operator ruling 2026-11-10T06:25:53Z) and is now deleted, not switched off:

* an instance that disables ``lab-overlay`` still builds the onex-api image and
  delivers the pin (on the old agent the whole settle was skipped);
* an instance that keeps ``lab-overlay`` gets the same build then pin, and no
  apply;
* an instance that disables ``onex-api-pin`` builds and delivers nothing;
* the settle runs host ``docker`` commands only, one of them the onex-api build
  from this job's captured runtime image, and writes no apply record;
* a capture taken for another job is never built on, and the delivery is still
  restricted to the archived lineage.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.executor import EnumInstancePhase, EnumRuntimeLane
from deploy_agent.job_state import JobStore
from deploy_agent.lab_overlay import API_IMAGE_NAME, API_REVISION_LABEL

from .test_lab_overlay_omn18200 import (
    MANIFEST_SHA,
    FakeRunner,
    _applier,
    captured,
)
from .test_settle_releases_dev_lane_omn19501 import (
    INFRA_SHA,
    OVERLAY_SHA,
    WAIT_SECONDS,
    _Applier,
    _cmd,
    _Executor,
)

pytestmark = [pytest.mark.unit, pytest.mark.real_settle_pool]


#: Every terminal payload the agent under test published, newest last.
PUBLISHED: list[dict[str, Any]] = []


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[DeployAgent]:
    lock_dir = tmp_path / "lane-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(lock_dir))
    monkeypatch.delenv("ONEX_LANE_LOCK_HELD", raising=False)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy.lock")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", True)
    PUBLISHED.clear()
    monkeypatch.setattr(
        agent_mod, "publish_result", lambda payload, config: PUBLISHED.append(payload)
    )
    deploy_agent = DeployAgent(skip_self_update=True)
    deploy_agent.job_store = JobStore(tmp_path / "jobs")
    try:
        yield deploy_agent
    finally:
        deploy_agent._settle_pool.shutdown(wait=True)


def _phases_off(*off: EnumInstancePhase) -> Any:
    def _lane_runs_phase(lane: EnumRuntimeLane, phase: EnumInstancePhase) -> bool:
        return phase not in off

    return _lane_runs_phase


def _roll(
    agent: DeployAgent, applier: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], dict[str, Any]]:
    events: list[str] = []
    if isinstance(applier, _Applier):
        applier.events = events
        applier.gate.set()
    monkeypatch.setattr(agent, "executor", _Executor(events))
    monkeypatch.setattr(agent, "_lab_overlay_applier", lambda: applier)
    job = _cmd()
    agent.job_store.accept(job.correlation_id, job.model_dump(mode="json"))
    agent._execute_command(job)
    agent.drain_settle(timeout=WAIT_SECONDS)
    assert [p["correlation_id"] for p in PUBLISHED] == [str(job.correlation_id)]
    return events, PUBLISHED[0]


def _gated_applier() -> _Applier:
    return _Applier([], threading.Event())


def test_lab_overlay_disabled_still_builds_and_delivers_the_pin(
    agent: DeployAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED on the old agent: ``lab-overlay`` off skipped the whole settle, so
    neither the image build nor the pin ran."""
    monkeypatch.setattr(
        agent_mod, "lane_runs_phase", _phases_off(EnumInstancePhase.LAB_OVERLAY)
    )
    events, payload = _roll(agent, _gated_applier(), monkeypatch)
    assert "build:start" in events and "pin" in events, events
    assert events.index("build:end") < events.index("pin"), events
    assert payload["onex_api_delivery"]["result"] == "WRITTEN", payload


def test_lab_overlay_enabled_runs_the_same_build_then_pin_and_no_apply(
    agent: DeployAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instance that still lists ``lab-overlay``: build, then pin. RED on the
    old agent, which called the applier's apply here instead of a build."""
    events, payload = _roll(agent, _gated_applier(), monkeypatch)
    assert (
        events.index("build:start") < events.index("build:end") < events.index("pin")
    ), events
    assert payload["onex_api_delivery"]["result"] == "WRITTEN", payload


def test_pin_phase_disabled_builds_and_delivers_nothing(
    agent: DeployAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An instance that runs no onex-api settles nothing; it still publishes."""
    monkeypatch.setattr(
        agent_mod, "lane_runs_phase", _phases_off(EnumInstancePhase.ONEX_API_PIN)
    )
    events, payload = _roll(agent, _gated_applier(), monkeypatch)
    assert not {"capture", "build:start", "pin"} & set(events), events
    assert payload["onex_api_delivery"] is None, payload


def test_the_settle_runs_host_docker_commands_only(
    agent: DeployAgent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real applier through a recording runner. RED on the old agent, whose
    apply ran cluster, content-store and apply-script commands here."""
    runner = FakeRunner()
    source = tmp_path / "overlay-source"
    source.mkdir()
    applier = _applier(tmp_path, source, runner)
    _events, payload = _roll(agent, applier, monkeypatch)

    assert {call[0] for call in runner.calls} == {"docker"}, runner.calls
    api_builds = [
        call
        for call in runner.argv_containing("docker build")
        if call[call.index("-t") + 1].startswith(f"{API_IMAGE_NAME}:")
    ]
    assert len(api_builds) == 1, runner.calls
    argv = api_builds[0]
    assert argv[argv.index("-t") + 1].startswith(
        f"{API_IMAGE_NAME}:{MANIFEST_SHA[:8]}-"
    )
    assert f"{API_REVISION_LABEL}={MANIFEST_SHA}" in argv
    runtime_tag = runner.argv_containing("docker tag")[0][-1]
    assert f"RUNTIME_IMAGE={runtime_tag}" in argv
    assert not list((tmp_path / "state" / "lab-overlay").glob("*.json"))
    assert payload["onex_api_delivery"]["result"] == "WRITTEN", payload


def test_a_capture_for_another_job_is_never_built_on(
    agent: DeployAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handler refuses another job's runtime image; the delivery still names
    the archived lineage and nothing else."""
    applier = _gated_applier()
    builds: list[list[str]] = []

    def _run(argv: Any, **kwargs: Any) -> Any:
        builds.append(list(argv))
        raise AssertionError("a build ran on another job's capture")

    monkeypatch.setattr(applier, "_run", _run)
    monkeypatch.setattr(
        applier,
        "capture_compose_inputs",
        lambda *, sha, stamp: captured("c" * 40, stamp),
    )
    events, payload = _roll(agent, applier, monkeypatch)

    assert builds == []
    assert "pin" in events, events
    assert payload["onex_api_delivery"]["requested_sha"] == OVERLAY_SHA, payload
    assert OVERLAY_SHA != INFRA_SHA
