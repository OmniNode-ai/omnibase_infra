# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23521 -- the onex-api pin is delivered when the k3s apply is retired.

THE DEFECT THESE TESTS PIN
--------------------------
The k3s ``onex-lab`` apply is retired for the dev instance (operator ruling
2026-11-10T06:25:53Z): its cluster's control service is dead and the step failed
every roll. The declared switch is ``lab-overlay`` in the instance's
``disabled_phases``. The settle ran only when ``lab-overlay`` was enabled, so the
switch also skipped the onex-api pin delivery, which is the separate
``onex-api-pin`` phase and which the compose dev lane still needs.

* an instance that disables ``lab-overlay`` builds the onex-api image on its own
  and still delivers the pin, and runs no apply;
* an instance that keeps ``lab-overlay`` is unchanged: apply, then pin;
* an instance that disables ``onex-api-pin`` delivers nothing;
* the image build runs the one docker build and touches no k3s, kubectl or
  apply script.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.executor import EnumInstancePhase, EnumSelfUpdateBoundary
from deploy_agent.job_state import JobStore
from deploy_agent.lab_overlay import (
    API_IMAGE_NAME,
    API_REVISION_LABEL,
    LabOverlayRefusalError,
    ModelLabOverlayCapture,
)

from .test_lab_overlay_omn18200 import (
    MANIFEST_SHA,
    SHA,
    STAMP,
    FakeRunner,
    _applier,
)

pytestmark = [pytest.mark.unit, pytest.mark.real_settle_pool]

INFRA_SHA = "a" * 40
OVERLAY_SHA = "b" * 40

#: Bound on every wait in this file. A test that deadlocks must fail, not hang.
WAIT_SECONDS = 10.0


class _Executor:
    """The executor surface the deploy path touches, recording its calls."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.oneshot_reruns: list[object] = []
        self.cloud_migration_failure: str = ""
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []
        self.self_update_calls: list[EnumSelfUpdateBoundary] = []

    def preflight(self, **kwargs: object) -> None:
        return None

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        self.events.append(f"git_pull:{git_ref}")
        return INFRA_SHA

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None:
        return None

    def seed_infisical(self, **kwargs: object) -> None:
        return None

    def validate_llm_endpoint_env_contract(self) -> None:
        return None

    def rebuild_scope(self, *args: object, **kwargs: Any) -> list[str]:
        self.events.append(f"rebuild:{kwargs.get('git_ref')}")
        return ["omninode-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        return []

    def run_lane_mirror(self, **kwargs: object) -> None:
        return None

    def deliver_onex_api_pin(self, **kwargs: Any) -> dict[str, Any]:
        self.events.append("pin")
        return {"result": "WRITTEN", "tag_advanced": True, "recreated": True}

    def self_update(
        self,
        *,
        boundary: EnumSelfUpdateBoundary,
        skip: bool = False,
        on_before_reexec: Any = None,
    ) -> None:
        self.self_update_calls.append(boundary)


class _Applier:
    """A lab-overlay applier whose apply blocks until the test opens a gate."""

    def __init__(self, events: list[str], gate: threading.Event) -> None:
        self.events = events
        self.gate = gate
        self.running = threading.Event()
        self.manifest_sha: str | None = None
        self.captures: list[dict[str, str]] = []

    def capture_compose_inputs(self, *, sha: str, stamp: str) -> dict[str, str]:
        capture = {"sha": sha, "stamp": stamp}
        self.captures.append(capture)
        self.events.append("capture")
        return capture

    def build_onex_api_image(self, *, sha: str, stamp: str, capture: Any) -> str:
        self.events.append("build_api")
        self.manifest_sha = OVERLAY_SHA
        return OVERLAY_SHA

    def apply(
        self, *, sha: str, stamp: str, correlation_id: str, capture: Any = None
    ) -> Path:
        self.events.append("apply:start")
        self.manifest_sha = OVERLAY_SHA
        self.running.set()
        self.gate.wait(timeout=WAIT_SECONDS)
        self.running.clear()
        self.events.append("apply:end")
        return Path(f"/state/lab-overlay/{sha}.json")


def _cmd(ref: str = "dev") -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
        git_ref=ref,
    )


@pytest.fixture
def harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]]]:
    lock_dir = tmp_path / "lane-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(lock_dir))
    monkeypatch.delenv("ONEX_LANE_LOCK_HELD", raising=False)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy.lock")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", True)

    events: list[str] = []
    published: list[dict[str, Any]] = []

    def _publish(payload: dict[str, Any], config: object) -> bool:
        published.append(payload)
        events.append(f"publish:{payload.get('correlation_id')}")
        return True

    monkeypatch.setattr(agent_mod, "publish_result", _publish)
    gate = threading.Event()
    executor = _Executor(events)
    applier = _Applier(events, gate)
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = JobStore(tmp_path / "jobs")
    agent.executor = executor  # type: ignore[assignment]
    monkeypatch.setattr(agent, "_lab_overlay_applier", lambda: applier)
    agent.published = published  # type: ignore[attr-defined]
    try:
        yield agent, executor, applier, gate, events
    finally:
        gate.set()
        agent._settle_pool.shutdown(wait=True)


def _run_in_thread(fn: Any, *args: Any) -> threading.Thread:
    thread = threading.Thread(target=fn, args=args, daemon=True)
    thread.start()
    return thread



def _phases_off(*off: EnumInstancePhase) -> Any:
    def _lane_runs_phase(lane: EnumRuntimeLane, phase: EnumInstancePhase) -> bool:
        return phase not in off

    return _lane_runs_phase


def _roll(
    harness: tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]],
) -> list[str]:
    agent, _executor, _applier_double, gate, events = harness
    gate.set()
    job = _cmd()
    agent.job_store.accept(job.correlation_id, job.model_dump(mode="json"))
    agent._execute_command(job)
    agent.drain_settle(timeout=WAIT_SECONDS)
    published = [p["correlation_id"] for p in agent.published]  # type: ignore[attr-defined]
    assert str(job.correlation_id) in published, "the terminal result was not published"
    return events


def test_lab_overlay_disabled_still_delivers_the_pin(
    harness: tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED against the agent where ``lab-overlay`` off skipped the whole settle.

    There the resolver answered None, no settle ran, and the events held neither
    the image build nor the pin.
    """
    monkeypatch.setattr(
        agent_mod, "lane_runs_phase", _phases_off(EnumInstancePhase.LAB_OVERLAY)
    )
    events = _roll(harness)
    assert "apply:start" not in events and "apply:end" not in events, events
    assert "build_api" in events and "pin" in events, events
    assert events.index("build_api") < events.index("pin"), events


def test_lab_overlay_enabled_keeps_apply_then_pin(
    harness: tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]],
) -> None:
    """The unchanged path: the apply runs, then the pin, and no separate build."""
    events = _roll(harness)
    assert "build_api" not in events, events
    assert events.index("apply:start") < events.index("apply:end") < events.index("pin")


def test_pin_phase_disabled_delivers_nothing(
    harness: tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An instance that runs no onex-api builds and delivers nothing; it still publishes."""
    monkeypatch.setattr(
        agent_mod,
        "lane_runs_phase",
        _phases_off(EnumInstancePhase.LAB_OVERLAY, EnumInstancePhase.ONEX_API_PIN),
    )
    events = _roll(harness)
    assert not {"build_api", "apply:start", "pin"} & set(events), events


@pytest.fixture
def overlay_source(tmp_path: Path) -> Path:
    """A stand-in omninode_infra clone, as the repair-build test declares it."""
    source = tmp_path / "omninode_infra"
    (source / "k8s" / "onex-lab").mkdir(parents=True)
    (source / "k8s" / "onex-lab" / "apply_lab_lane.sh").write_text("#!/bin/sh\n")
    return source


def _capture(sha: str = SHA) -> ModelLabOverlayCapture:
    return ModelLabOverlayCapture(
        sha=sha,
        stamp=STAMP,
        runtime_image="runtime:img",
        infra_migrate_image="infra:img",
        pin_error=None,
        compose_omnimarket_version=None,
        compose_version_error=None,
    )


def test_build_onex_api_image_runs_only_the_api_docker_build(
    tmp_path: Path, overlay_source: Path
) -> None:
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)

    built_from = applier.build_onex_api_image(sha=SHA, stamp=STAMP, capture=_capture())

    assert built_from == MANIFEST_SHA
    assert applier.manifest_sha == MANIFEST_SHA
    builds = runner.argv_containing("docker build")
    assert len(builds) == 1, builds
    argv = builds[0]
    assert argv[argv.index("-t") + 1] == f"{API_IMAGE_NAME}:{MANIFEST_SHA[:8]}-{STAMP}"
    assert "RUNTIME_IMAGE=runtime:img" in argv
    assert f"{API_REVISION_LABEL}={MANIFEST_SHA}" in argv
    joined = [" ".join(call) for call in runner.calls]
    assert not [c for c in joined if "k3s" in c or "kubectl" in c], joined
    assert not [c for c in joined if "apply_lab_lane.sh" in c], joined
    assert not list((tmp_path / "state" / "lab-overlay").glob("*.json"))


def test_build_onex_api_image_refuses_another_jobs_capture(
    tmp_path: Path, overlay_source: Path
) -> None:
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)

    with pytest.raises(LabOverlayRefusalError):
        applier.build_onex_api_image(
            sha=SHA, stamp=STAMP, capture=_capture(sha="c" * 40)
        )

    assert not runner.argv_containing("docker build")
