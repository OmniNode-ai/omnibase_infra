# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The agent converges its lane's deps before it consumes anything (OMN-18692).

WHY THIS IS A STARTUP CONCERN AND NOT ONLY A DEPLOY CONCERN. The dev-lane
agent's control bus IS the dev lane's redpanda:
``deploy/deploy-agent-dev.service`` points its ``KAFKA_BOOTSTRAP_SERVERS`` at
that lane's own external listener. On 2026-09-18 the CORE phase removed that container, so the process
that would have repaired the lane could not reach its own commands: it
crash-looped on ``NoBrokersAvailable`` against the broker it had just
destroyed, and systemd gave up at 13:17:31Z with restart counter 7. Nothing in
the startup path read the lane before reaching for the broker, and the lane sat
broken until an operator ran ``up -d postgres redpanda valkey`` by hand 22
minutes later.

So the ordering asserted here is the whole fix, and it is asserted
BEHAVIOURALLY -- both the convergence and the consumer's construction record
themselves in one list, and the test reads the order out of that list. A
source-order assertion would pass against a build where the call had been
moved into a branch that never runs.

THE FENCE IS TESTED IN BOTH DIRECTIONS. Starting containers without an accepted
command is a lane mutation, and doing it on a governed lane would be an
unattributed one -- the thing the OMN-15243 raw-bypass signature set and the
OMN-15218 attribution interlock exist to refuse. So it runs only for an agent
whose declared lane set is exactly ``{dev}``, the lane the lane table calls a
fully mutable test platform.
"""

from __future__ import annotations

import asyncio
import socket
import time
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane

pytestmark = pytest.mark.unit

SHA = "b" * 40


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _RecordingExecutor:
    """An executor that records whether, and when, convergence was asked for."""

    def __init__(self, timeline: list[str], *, was_down: list[str]) -> None:
        self._timeline = timeline
        self._was_down = was_down
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        # OMN-18640: the terminal event now also carries what verification
        # recreated, so a fake executor has to declare it.
        self.verify_recreate: list[object] = []
        # OMN-18640: the terminal event now also carries what the deps leg
        # found before it acted, and the argv of every compose call, so a
        # fake executor has to declare both.
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        # OMN-18640 AC8: the agent publishes the executor's own probe
        # readings when its local list is empty, which is the case on
        # every job that failed verification.
        self.health_checks: list[object] = []
        self.converge_calls: list[EnumRuntimeLane] = []

    def converge_deps(
        self, *, lane: EnumRuntimeLane = EnumRuntimeLane.DEV
    ) -> tuple[bool, list[str]]:
        self._timeline.append("converge_deps")
        self.converge_calls.append(lane)
        return True, list(self._was_down)

    def self_update(self, *, boundary: Any, **kwargs: Any) -> None:
        pass


def _silent_consumer(timeline: list[str]) -> type:
    class _SilentConsumer:
        def __init__(self, **kwargs: Any) -> None:
            timeline.append("consumer_constructed")

        def poll_and_accept(self) -> tuple[None, None]:
            time.sleep(0.05)
            return None, None

        def close(self) -> None:
            pass

    return _SilentConsumer


async def _run_briefly(agent: DeployAgent) -> None:
    task = asyncio.create_task(agent.run())
    try:
        await asyncio.sleep(0.3)
    finally:
        agent._shutdown = True
        await asyncio.wait_for(task, timeout=10)


@pytest.fixture
def startup_seams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Boot the real ``run()`` with nothing that reaches the network or docker."""
    timeline: list[str] = []
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy.lock")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "jobs")
    monkeypatch.setattr(agent_mod, "HEALTH_PORT", _free_port())
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    monkeypatch.setattr(agent_mod, "record_loaded_code_sha", lambda _agent_dir: SHA)
    monkeypatch.setattr(agent_mod, "DeployConsumer", _silent_consumer(timeline))
    return timeline


@pytest.mark.asyncio
async def test_a_half_recreated_lane_converges_its_deps_before_the_consumer_exists_omn18692(
    startup_seams: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3: the deps are converged BEFORE anything is consumed.

    The falsifier is the ORDER, not the presence of the call: a convergence
    that ran after the consumer was built would have had to connect to the
    absent broker first, which is exactly the crash loop being removed.
    """
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    agent = DeployAgent(skip_self_update=True)
    executor = _RecordingExecutor(startup_seams, was_down=["redpanda", "valkey"])
    agent.executor = executor  # type: ignore[assignment]

    await _run_briefly(agent)

    assert "converge_deps" in startup_seams, (
        "the agent consumed without ever reading its lane -- the 2026-09-18 "
        "startup path"
    )
    assert startup_seams.index("converge_deps") < startup_seams.index(
        "consumer_constructed"
    )
    assert executor.converge_calls == [EnumRuntimeLane.DEV]


@pytest.mark.asyncio
async def test_a_converged_lane_still_gets_read_and_still_starts(
    startup_seams: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy lane costs one read; the agent starts either way.

    The convergence itself decides whether to act (it reads ``docker compose
    ps`` first and returns early on a running lane -- asserted in
    ``test_executor_deps_recreate_omn18692.py``), so the agent calls it
    unconditionally rather than trying to predict the answer.
    """
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    agent = DeployAgent(skip_self_update=True)
    executor = _RecordingExecutor(startup_seams, was_down=[])
    agent.executor = executor  # type: ignore[assignment]

    await _run_briefly(agent)

    assert executor.converge_calls == [EnumRuntimeLane.DEV]
    assert "consumer_constructed" in startup_seams


@pytest.mark.asyncio
async def test_the_convergence_is_fenced_to_a_dev_only_agent(
    startup_seams: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent that also carries a governed lane starts NOTHING on its own.

    ``deploy-agent.service`` -- the prod-pinned sibling unit -- and any agent
    whose fence admits ``stability-test`` must not bring containers up without
    an accepted command. That would be a lane mutation with no attribution
    record, which is the shape the OMN-15218 interlock exists to refuse, and a
    recovery that had to break a deploy gate to run would be a worse defect
    than the one it repairs.
    """
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev,stability-test")
    agent = DeployAgent(skip_self_update=True)
    executor = _RecordingExecutor(startup_seams, was_down=["redpanda"])
    agent.executor = executor  # type: ignore[assignment]

    await _run_briefly(agent)

    assert executor.converge_calls == []
    assert "converge_deps" not in startup_seams
    assert "consumer_constructed" in startup_seams


@pytest.mark.asyncio
async def test_a_raising_convergence_does_not_stop_the_agent_starting(
    startup_seams: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ERROR CHAIN for the startup half: the repair may fail; startup may not.

    An agent that refused to start because its lane was broken is an agent that
    cannot report that its lane is broken -- it has a job store to recover, a
    health surface to serve and pending publishes to retry, all of which are
    how a reader finds out. So the convergence is non-fatal, and that is
    asserted rather than assumed.
    """
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")

    class _RaisingExecutor(_RecordingExecutor):
        def converge_deps(
            self, *, lane: EnumRuntimeLane = EnumRuntimeLane.DEV
        ) -> tuple[bool, list[str]]:
            self._timeline.append("converge_deps")
            raise RuntimeError("Cannot connect to the Docker daemon")

    agent = DeployAgent(skip_self_update=True)
    agent.executor = _RaisingExecutor(startup_seams, was_down=[])  # type: ignore[assignment]

    await _run_briefly(agent)

    assert startup_seams.index("converge_deps") < startup_seams.index(
        "consumer_constructed"
    )
