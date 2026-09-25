# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19501 -- the k3s lab-overlay settle stops holding the .201 dev lane.

THE MEASUREMENT
---------------
On 2026-09-24 the dev deploy agent completed 30 jobs. After each compose verdict
it kept the job thread AND the dev lane lock through the k3s ``onex-lab``
overlay apply and the onex-api pin delivery: 29 settle stages, median 6.0 min
(5.3 to 13.0), against a median compose job of 15.7 min. The next command
waited the whole time, although the overlay mutates a different surface with
its own receipt lane (``onex-lab-k3s``) and only the pin recreate touches the
compose project.

THE CHANGE THESE TESTS PIN
--------------------------
* the compose job ends at its verdict: the dev lane lock and the job thread are
  released, and the settle runs on its own single-worker pool;
* the overlay runs under its own surface lock, never the compose lane's;
* the onex-api pin recreate RE-ACQUIRES the compose lane lock, because it is a
  mutation of the compose project;
* a self-update re-exec waits for the settle in flight, so a re-exec never
  kills an overlay apply or a pin recreate halfway.

The design was model-checked first: OMN-19421's ``LabReconcile.tla`` extended
with the settle worker, verdict and TLC logs on OMN-19501.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.executor import EnumSelfUpdateBoundary, lane_config_for
from deploy_agent.job_state import EnumJobSettlingStage, JobStore

pytestmark = [pytest.mark.unit, pytest.mark.real_settle_pool]

INFRA_SHA = "a" * 40
OVERLAY_SHA = "b" * 40
DEV_PROJECT = lane_config_for(EnumRuntimeLane.DEV).compose_project

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


# --------------------------------------------------------------------------- #
# AC1 -- a settling job does not make the next dev command wait               #
# --------------------------------------------------------------------------- #
def test_settle_does_not_block_the_next_compose_job(
    harness: tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]],
) -> None:
    """RED against the tree where the settle ran on the job thread.

    Job 1's overlay apply is held open. On the old tree ``_execute_command(job1)``
    does not return until the apply does, so the job thread (a pool of one) and
    the dev lane lock both stay taken and job 2 cannot start. On the new tree job
    1's command returns at its verdict and job 2 runs its whole compose phase
    while job 1's overlay is still applying.
    """
    agent, _executor, applier, gate, events = harness
    job1, job2 = _cmd("ref-1"), _cmd("ref-2")
    agent.job_store.accept(job1.correlation_id, job1.model_dump(mode="json"))

    first = _run_in_thread(agent._execute_command, job1)
    assert applier.running.wait(WAIT_SECONDS), "job 1 never reached its overlay"
    first.join(WAIT_SECONDS)
    assert not first.is_alive(), (
        "job 1's command did not return while its k3s overlay was still "
        "applying: the settle is still holding the job thread"
    )

    agent.job_store.accept(job2.correlation_id, job2.model_dump(mode="json"))
    second = _run_in_thread(agent._execute_command, job2)
    second.join(WAIT_SECONDS)
    assert not second.is_alive(), "job 2 could not run while job 1 settled"
    assert "rebuild:ref-2" in events
    assert applier.running.is_set(), "job 1's overlay was still meant to be open"
    job2_record = agent.job_store.load(job2.correlation_id)
    assert job2_record is not None and job2_record.status == "success"

    gate.set()
    agent.drain_settle(timeout=WAIT_SECONDS)
    published = [p["correlation_id"] for p in agent.published]  # type: ignore[attr-defined]
    assert str(job1.correlation_id) in published
    assert str(job2.correlation_id) in published


def test_settle_does_not_block_the_consumer_busy_check(tmp_path: Path) -> None:
    """The consumer's ``busy`` gate reads ``has_active_job``. A record whose
    verdict is written and whose settle stage is ``lab_overlay`` is terminal,
    so it must never read as busy -- the regression guard for AC1's other half.
    """
    store = JobStore(tmp_path / "jobs")
    settling = _cmd()
    store.accept(settling.correlation_id, settling.model_dump(mode="json"))
    store.complete(
        settling.correlation_id,
        status="success",
        settling_stage=EnumJobSettlingStage.LAB_OVERLAY,
    )
    assert store.has_active_job() is False
    assert store.load_active() is None


# --------------------------------------------------------------------------- #
# AC2 -- the pin recreate runs only while the compose lane lock is held        #
# --------------------------------------------------------------------------- #
def test_onex_api_pin_under_lane_lock(
    harness: tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock double records acquire and release order around the whole job.

    Required order: the job's dev-lane hold ends BEFORE the overlay starts; the
    overlay runs under the k3s surface lock and not the dev lane lock; the pin
    runs inside a SECOND, separate dev-lane hold. RED against the tree where the
    overlay and the pin ran inside the job's single hold.
    """
    agent, executor, applier, gate, events = harness
    held: list[str] = []

    @contextmanager
    def _recording_lock(project: str, **kwargs: Any) -> Iterator[None]:
        events.append(f"acquire:{project}")
        held.append(project)
        try:
            yield
        finally:
            held.remove(project)
            events.append(f"release:{project}")

    monkeypatch.setattr(agent_mod, "lane_lock", _recording_lock)
    original_apply = applier.apply

    def _apply_checking_locks(**kwargs: Any) -> Path:
        assert DEV_PROJECT not in held, (
            f"the k3s overlay ran while the dev lane lock was held: {held}"
        )
        assert agent_mod.LAB_OVERLAY_SURFACE_LOCK in held, held
        return original_apply(**kwargs)

    applier.apply = _apply_checking_locks  # type: ignore[method-assign]
    original_pin = executor.deliver_onex_api_pin

    def _pin_checking_locks(**kwargs: Any) -> dict[str, Any]:
        assert held == [DEV_PROJECT], (
            f"the onex-api pin recreate ran outside the dev lane lock: {held}"
        )
        return original_pin(**kwargs)

    executor.deliver_onex_api_pin = _pin_checking_locks  # type: ignore[method-assign]

    gate.set()
    job = _cmd()
    agent.job_store.accept(job.correlation_id, job.model_dump(mode="json"))
    agent._execute_command(job)
    agent.drain_settle(timeout=WAIT_SECONDS)

    order = [
        e
        for e in events
        if e.split(":")[0] in {"acquire", "release", "apply", "pin", "capture"}
    ]
    dev_acquires = [i for i, e in enumerate(order) if e == f"acquire:{DEV_PROJECT}"]
    dev_releases = [i for i, e in enumerate(order) if e == f"release:{DEV_PROJECT}"]
    assert len(dev_acquires) == 2 and len(dev_releases) == 2, order
    capture, apply_start, pin = (
        order.index("capture"),
        order.index("apply:start"),
        order.index("pin"),
    )
    # The capture reads compose-lane state, so it is inside the job's hold.
    assert dev_acquires[0] < capture < dev_releases[0], order
    # The overlay starts after the job's hold ends.
    assert dev_releases[0] < apply_start, order
    # The pin is inside the second hold, after the overlay.
    assert apply_start < dev_acquires[1] < pin < dev_releases[1], order


def test_the_overlay_reads_the_capture_taken_under_the_lane_lock(
    harness: tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]],
) -> None:
    """Model condition OverlayPromotesOwnBuild: the overlay gets the capture.

    The applier's ``capture_compose_inputs`` is called for THIS job's sha, and
    the same object is handed to ``apply``. An overlay that read
    ``runtime:latest`` when it ran would promote the NEXT job's build as this
    one's once the lane is released at the verdict.
    """
    agent, _executor, applier, gate, _events = harness
    seen: list[Any] = []
    original_apply = applier.apply

    def _apply(**kwargs: Any) -> Path:
        seen.append(kwargs.get("capture"))
        return original_apply(**kwargs)

    applier.apply = _apply  # type: ignore[method-assign]
    gate.set()
    job = _cmd()
    agent.job_store.accept(job.correlation_id, job.model_dump(mode="json"))
    agent._execute_command(job)
    agent.drain_settle(timeout=WAIT_SECONDS)

    assert len(applier.captures) == 1
    assert applier.captures[0]["sha"] == INFRA_SHA
    assert seen == [applier.captures[0]]


def test_the_terminal_event_waits_for_the_pin_and_carries_it(
    harness: tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]],
) -> None:
    """The terminal event keeps OMN-18572's shape: it is published after the
    pin delivery and carries its verdict. Only WHO waits for it changed."""
    agent, _executor, _applier, gate, events = harness
    gate.set()
    job = _cmd()
    agent.job_store.accept(job.correlation_id, job.model_dump(mode="json"))
    agent._execute_command(job)
    agent.drain_settle(timeout=WAIT_SECONDS)

    assert events.index("pin") < events.index(f"publish:{job.correlation_id}")
    payload = agent.published[0]  # type: ignore[attr-defined]
    assert payload["onex_api_delivery"]["result"] == "WRITTEN"
    record = agent.job_store.load(job.correlation_id)
    assert record is not None
    assert record.settling_stage is None
    assert record.onex_api_delivery is not None


# --------------------------------------------------------------------------- #
# A re-exec never kills a settle halfway                                       #
# --------------------------------------------------------------------------- #
def test_a_reexec_waits_for_the_settle_in_flight(
    harness: tuple[DeployAgent, _Executor, _Applier, threading.Event, list[str]],
) -> None:
    """``self_update`` calls ``on_before_reexec`` immediately before replacing
    the process image. The agent must hand every boundary a callback that waits
    for the settle in flight, or an idle-boundary re-exec lands in the middle of
    ``k3s ctr images import`` and the sha's k3s record is never written."""
    agent, executor, applier, gate, events = harness
    job = _cmd()
    agent.job_store.accept(job.correlation_id, job.model_dump(mode="json"))
    agent._execute_command(job)
    assert applier.running.wait(WAIT_SECONDS)

    callbacks: list[Any] = []

    def _self_update(
        *,
        boundary: EnumSelfUpdateBoundary,
        skip: bool = False,
        on_before_reexec: Any = None,
    ) -> None:
        callbacks.append(on_before_reexec)

    executor.self_update = _self_update  # type: ignore[method-assign]
    agent._self_update_post_terminal()
    assert callbacks and callbacks[0] is not None, (
        "the post-terminal boundary can re-exec with no wait for the settle"
    )

    waited = threading.Event()

    def _reexec_now() -> None:
        callbacks[0]()
        waited.set()

    reexec = _run_in_thread(_reexec_now)
    time.sleep(0.3)
    assert not waited.is_set(), "the re-exec did not wait for the overlay apply"
    gate.set()
    reexec.join(WAIT_SECONDS)
    assert waited.is_set()
    assert "apply:end" in events
