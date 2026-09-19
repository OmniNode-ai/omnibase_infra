# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18636 -- the HTTP surface must answer while a deploy phase is executing.

THE DEFECT THESE TESTS PIN
--------------------------
``deploy_agent.health.create_health_app`` built an aiohttp application, and
``DeployAgent.run`` started it on the loop that ``__main__`` created. The same
coroutine then ran the poll loop, and every executor phase reached from it was a
plain synchronous ``subprocess.run`` -- not awaited, not handed to a thread. For
the length of every phase the loop thread sat inside ``_communicate``'s selector
poll, nothing called ``accept()``, and the agent answered nothing.

Measured 2026-09-17, lane ``deploy-agent-http-hang-diag-2105`` (TERMINAL row
22:10:00Z), 121 samples against the live ``.201`` dev-lane agent: 107 returned
curl code ``000`` and 14 returned ``200``. The listen backlog climbed 26 -> 129
against a limit of 128, after which the kernel silently drops SYNs -- so the
unreachability outlived the call that caused it. Downstream,
``check_dev_lane_staleness.py`` could not read ``deployed_revision`` off
``/job/{correlation_id}``, the compose-dev lab-pass receipt read
``deployed_revision: indeterminate`` for a lane that HAD converged, and Operating
Rule 24(b) closed delivery to staging for a good sha.

WHY THE PROBE RUNS ON ITS OWN THREAD
------------------------------------
A probe driven from the test's own coroutine would be measuring the loop from
inside the loop: when the loop is blocked the probe cannot issue its request
either, so a failure would be indistinguishable from the test itself being
starved. The reader this defect actually broke is a ``curl`` in a CI container --
a separate process on the other side of a socket -- so the probe here is a
``urllib`` request on a plain ``threading.Thread``, which is the closest
in-process analogue: it is scheduled by the OS, not by the loop under test, and
it observes exactly what the kernel's accept queue does.

WHAT IS NOT ASSERTED, DELIBERATELY
----------------------------------
No test here reads the source for ``run_in_executor`` or counts ``async def``
keywords. A text assertion passes against a tree where the behaviour regressed
for some other reason and fails against a tree where the behaviour is right but
the mechanism moved. The surface is EXERCISED: a phase blocks, and an external
thread asks the agent a question while it does.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import JOB_POOL_MAX_WORKERS, DeployAgent
from deploy_agent.events import (
    EnumRuntimeLane,
    EnumSelfUpdateBoundary,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.job_state import JobState, JobStore, describe_interruption

pytestmark = pytest.mark.unit

SHA = "a" * 40

#: How long the stub phase holds, standing in for a ``subprocess.run``. Long
#: enough that a probe issued at phase entry, and its whole timeout, land
#: strictly inside the phase -- otherwise a pass could mean "the phase had
#: already finished" rather than "the loop stayed free".
BLOCK_SECONDS = 3.0

#: The bound the probe asserts, and the bound the RED case must exceed. The
#: receipt reader's own curl uses ``--max-time 2``; this is tighter, so a pass
#: here is a pass there. It is also comfortably below BLOCK_SECONDS, which is
#: what makes the RED case unambiguous: on a blocked loop the request cannot be
#: served for another 3 s, so it times out here rather than answering late.
PROBE_TIMEOUT_SECONDS = 1.0

#: How long the test is willing to wait for the agent to reach the blocking
#: phase at all. Generous, because it bounds a scheduling wait rather than the
#: behaviour under test; exceeding it is a hung test, not a slow one.
PHASE_ENTRY_TIMEOUT_SECONDS = 30.0


def _free_port() -> int:
    """A port nothing is listening on, resolved by binding and releasing one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ProbeResult:
    """One external HTTP observation: what came back, and when it came back."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.status: int | None = None
        self.body: dict[str, Any] | None = None
        self.error: str | None = None
        self.elapsed: float = float("inf")
        self.mid_phase: bool = False

    def __repr__(self) -> str:  # pragma: no cover - assertion messages only
        return (
            f"<probe {self.url} status={self.status} error={self.error} "
            f"elapsed={self.elapsed:.3f}s mid_phase={self.mid_phase}>"
        )


def _probe_once(url: str, still_running: threading.Event) -> _ProbeResult:
    """One bounded GET, recording whether the phase was still running.

    ``still_running`` is SET for the duration of the blocking phase and cleared
    when it ends, so ``mid_phase`` records the fact the assertion needs: that the
    answer was obtained while the agent was busy, not after it went idle.
    """
    result = _ProbeResult(url)
    # No proxy, ever: an ambient http_proxy on a runner would send this probe
    # somewhere other than the agent under test and the result would be about
    # the proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    try:
        with opener.open(url, timeout=PROBE_TIMEOUT_SECONDS) as response:
            payload = response.read()
            result.status = int(response.status)
            result.body = json.loads(payload)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        result.error = repr(exc)
    result.elapsed = time.monotonic() - started
    result.mid_phase = still_running.is_set()
    return result


class _BlockingExecutor:
    """A deploy whose first phase blocks its thread for ``BLOCK_SECONDS``.

    ``time.sleep`` rather than a real ``subprocess.run``: the property under test
    is that the calling thread does not return to the event loop for the
    duration, and a sleep holds the thread exactly as a subprocess wait does
    while needing no docker, no network and no host. The seven real sites are in
    ``executor.py``; which blocking primitive they use is not what broke.
    """

    def __init__(self, entered: threading.Event, running: threading.Event) -> None:
        self.calls: list[str] = []
        self.boundaries: list[str] = []
        self._entered = entered
        self._running = running
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        # OMN-18692: the agent reads this when it builds the terminal event,
        # so a double that omits it no longer models the object it replaces.
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

    def preflight(self, **kwargs: Any) -> None:
        self.calls.append("preflight")
        on_phase_update = kwargs["on_phase_update"]
        on_phase_update(Phase.PREFLIGHT, PhaseStatus.IN_PROGRESS)
        self._running.set()
        self._entered.set()
        time.sleep(BLOCK_SECONDS)
        self._running.clear()
        on_phase_update(Phase.PREFLIGHT, PhaseStatus.SUCCESS)

    def git_pull(self, git_ref: str, **kwargs: Any) -> str:
        self.calls.append("git_pull")
        return SHA

    def compose_gen(self, bundles: list[str], **kwargs: Any) -> None:
        self.calls.append("compose_gen")

    def seed_infisical(self, **kwargs: Any) -> None:
        self.calls.append("seed_infisical")

    def validate_llm_endpoint_env_contract(self) -> None:
        self.calls.append("validate_llm_endpoint_env_contract")

    def rebuild_scope(self, *args: Any, **kwargs: Any) -> list[str]:
        self.calls.append("rebuild_scope")
        return ["omninode-runtime"]

    def verify(self, **kwargs: Any) -> list[object]:
        self.calls.append("verify")
        return []

    def self_update(self, *, boundary: Any, **kwargs: Any) -> None:
        self.boundaries.append(str(boundary.value))


class _OneShotConsumer:
    """Serves one command, accepts it into the store, then polls empty forever.

    Constructed by ``DeployAgent.run`` itself, so the class attribute is how a
    test hands it the command -- the same shape ``_FakeApplier`` uses in
    ``test_lab_overlay_build_order_omn18545.py``.
    """

    command: ModelRebuildRequested | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.job_store: JobStore = kwargs["job_store"]
        self.closed = False
        self.polls = 0
        self._served = False

    def poll_and_accept(
        self,
    ) -> tuple[ModelRebuildRequested | None, str | None]:
        self.polls += 1
        command = _OneShotConsumer.command
        if self._served or command is None:
            # The real poll blocks for up to its 1000 ms kafka timeout. A short
            # sleep keeps this one a blocking call too -- an instant return
            # would make the idle branch of the loop unrepresentative of the
            # thing being fixed.
            time.sleep(0.05)
            return None, None
        self._served = True
        self.job_store.accept(command.correlation_id, command.model_dump(mode="json"))
        return command, None

    def close(self) -> None:
        self.closed = True


def _dev_command() -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )


def _make_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    port: int,
    executor: _BlockingExecutor,
) -> DeployAgent:
    """A DeployAgent wired to a temp state dir, a free port and a fake bus.

    Every stub here is a seam the agent already has. The lab overlay is switched
    OFF because this test is about the HTTP surface, not the k3s apply, and the
    applier shells out to containerd; its own behaviour is asserted in
    ``test_lab_overlay_build_order_omn18545.py``.
    """
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy.lock")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "jobs")
    monkeypatch.setattr(agent_mod, "HEALTH_PORT", port)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    monkeypatch.setattr(agent_mod, "DeployConsumer", _OneShotConsumer)
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: True)
    monkeypatch.setattr(agent_mod, "record_loaded_code_sha", lambda _agent_dir: SHA)

    agent = DeployAgent(skip_self_update=True)
    agent.executor = executor  # type: ignore[assignment]
    return agent


# --------------------------------------------------------------------------- #
# AC1 + AC2 + AC3 -- the surface answers while a phase blocks                  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_health_and_job_answer_within_the_bound_while_a_phase_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measurement, reproduced in a test.

    RED against the tree where ``_execute_command`` ran on the event loop: the
    3 s phase holds the loop thread, ``accept()`` is never called, and both
    probes come back as a 1 s timeout with no status -- the ``000`` the live
    sampling recorded 107 times. GREEN once the phase runs on the job thread:
    both answer in well under the bound, while the phase is still running.

    ``/job`` is probed as well as ``/health`` because it is the endpoint the
    defect actually cost us. ``check_dev_lane_staleness.py`` reads
    ``deployed_revision`` from it, and a job that cannot be read DURING
    convergence is exactly the ``indeterminate`` receipt verdict rule 24(b)
    refuses delivery on.
    """
    entered = threading.Event()
    running = threading.Event()
    port = _free_port()
    executor = _BlockingExecutor(entered, running)
    command = _dev_command()
    monkeypatch.setattr(_OneShotConsumer, "command", command, raising=False)

    agent = _make_agent(tmp_path, monkeypatch, port=port, executor=executor)

    probes: dict[str, _ProbeResult] = {}

    def _probe_worker() -> None:
        if not entered.wait(timeout=PHASE_ENTRY_TIMEOUT_SECONDS):
            return
        probes["health"] = _probe_once(f"http://127.0.0.1:{port}/health", running)
        probes["job"] = _probe_once(
            f"http://127.0.0.1:{port}/job/{command.correlation_id}", running
        )

    # The probe thread is started BEFORE the agent, and it waits on an event the
    # phase itself sets, so nothing about it depends on the loop getting a turn.
    # This ordering is load-bearing rather than stylistic: a probe started after
    # an `await` that needs the loop cannot run until the loop is free, which on
    # the defective tree is only ever AFTER the phase — so the test would fail
    # for the true reason but report the wrong one.
    prober = threading.Thread(target=_probe_worker, name="omn18636-probe")
    prober.start()
    run_task = asyncio.create_task(agent.run())
    try:
        await asyncio.to_thread(prober.join, PHASE_ENTRY_TIMEOUT_SECONDS + 30.0)
    finally:
        agent._shutdown = True
        await asyncio.wait_for(run_task, timeout=60.0)

    assert not prober.is_alive(), "the probe thread never finished"
    assert set(probes) == {"health", "job"}, (
        f"the blocking phase was never entered within "
        f"{PHASE_ENTRY_TIMEOUT_SECONDS}s: {executor.calls}"
    )

    health = probes["health"]
    assert health.mid_phase, (
        "the health probe landed after the phase ended, so it proves nothing "
        f"about a blocked loop: {health}"
    )
    assert health.error is None, (
        "GET /health did not answer while a deploy phase was executing -- this "
        "is the measured defect: the event loop is inside the phase, nothing "
        f"calls accept(), and the request times out. {health}"
    )
    assert health.status == 200, f"unexpected health status: {health}"
    assert health.elapsed < PROBE_TIMEOUT_SECONDS, (
        f"GET /health answered later than the {PROBE_TIMEOUT_SECONDS}s bound the "
        f"receipt reader allows: {health}"
    )
    assert health.body is not None
    assert health.body["state"] == "deploying", (
        f"the agent must report itself deploying while it deploys: {health.body}"
    )
    assert health.body["active_job"] is not None, (
        "a health payload served mid-deploy that names no active job is the "
        f"'no job in progress is not idle' confusion this ticket started in: "
        f"{health.body}"
    )

    job = probes["job"]
    assert job.mid_phase, f"the job probe landed after the phase ended: {job}"
    assert job.error is None, (
        "GET /job/{correlation_id} did not answer during convergence, which is "
        f"the read check_dev_lane_staleness.py makes: {job}"
    )
    assert job.status == 200, f"unexpected job status: {job}"
    assert job.elapsed < PROBE_TIMEOUT_SECONDS, f"job read too slow: {job}"
    assert job.body is not None
    assert job.body["status"] == "in_progress", (
        f"an in-flight job must read as in_progress, not as a terminal or "
        f"absent one: {job.body}"
    )
    assert job.body["current_phase"] == Phase.PREFLIGHT.value, (
        f"the reader must be able to see WHICH phase is running: {job.body}"
    )


# --------------------------------------------------------------------------- #
# AC6 -- a responsive surface must never cost the deploy                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_the_job_still_completes_with_its_phases_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The labelled fail-closed criterion, exercised.

    The failure mode a health fix invites is keeping the surface responsive by
    bounding, interrupting or abandoning the work behind it. A deploy killed to
    answer a health check manufactures exactly the false FAIL receipt this ticket
    exists to remove, so the job that outran every probe above must still reach
    its own terminal state with every phase recorded.
    """
    entered = threading.Event()
    running = threading.Event()
    port = _free_port()
    executor = _BlockingExecutor(entered, running)
    command = _dev_command()
    monkeypatch.setattr(_OneShotConsumer, "command", command, raising=False)

    agent = _make_agent(tmp_path, monkeypatch, port=port, executor=executor)

    run_task = asyncio.create_task(agent.run())
    try:
        await asyncio.to_thread(entered.wait, PHASE_ENTRY_TIMEOUT_SECONDS)
        # Ask for shutdown WHILE the phase is running, and the job must finish
        # anyway. Nothing here is allowed to cut it short.
        agent._shutdown = True
        await asyncio.wait_for(run_task, timeout=90.0)
    finally:
        agent._shutdown = True

    job = agent.job_store.load(command.correlation_id)
    assert job is not None, "the job record vanished"
    assert job.status == "success", (
        f"the deploy did not reach its own terminal state: {job.status} {job.errors}"
    )
    assert executor.calls[0] == "preflight"
    assert "verify" in executor.calls, (
        f"the deploy was curtailed before its verification phase: {executor.calls}"
    )
    assert job.phase_results[Phase.PREFLIGHT] == PhaseStatus.SUCCESS, (
        f"the blocking phase's own verdict was lost: {job.phase_results}"
    )
    assert job.phase_results[Phase.PUBLISH] == PhaseStatus.SUCCESS, (
        f"the terminal result was never published: {job.phase_results}"
    )


# --------------------------------------------------------------------------- #
# The concurrency contract the offload must not change                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_the_job_pool_runs_exactly_one_job_at_a_time() -> None:
    """One worker, and the number is the contract.

    Moving the work off the loop must not buy concurrency the rest of the design
    refuses: one compose project, one clone, one image cache, one lane lock, and
    a consumer that rejects a second command as ``busy``. A pool wider than one
    would let a publish retry or an idle self-update run alongside a deploy, and
    the self-update boundaries are defined by NOT doing that.
    """
    assert JOB_POOL_MAX_WORKERS == 1, (
        "widening the job pool is a change to the agent's concurrency contract, "
        "not a tuning knob -- see the constant's own note"
    )


@pytest.mark.asyncio
async def test_offloaded_calls_run_off_the_loop_and_on_one_shared_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two facts in one run: not the loop's thread, and the same thread twice.

    The first is the fix. The second is the serialization the fix must preserve
    -- two submissions landing on one worker means they cannot overlap, which is
    what keeps a publish retry from running inside a deploy.
    """
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "jobs")
    agent = DeployAgent(skip_self_update=True)
    loop_thread = threading.get_ident()

    first = await agent._offload(threading.get_ident)
    second = await agent._offload(threading.get_ident)

    assert first != loop_thread, (
        "the offloaded call ran on the event loop thread, which is the defect"
    )
    assert first == second, (
        "two offloaded calls ran on different threads, so the pool is wider "
        "than one and deploys are no longer serialized"
    )
    agent._job_pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_the_self_update_boundaries_still_fire_from_the_offloaded_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A boundary reached through a worker thread is still a boundary.

    The self-update boundaries re-exec, and they now do so from the job thread.
    ``os.execv`` from a non-main thread is defined -- POSIX terminates every
    other thread and the caller becomes the new image's initial thread -- but
    the boundary has to be REACHED before that matters, and the offload is a new
    seam between the poll loop and it. This asserts the post-terminal boundary
    still fires exactly once, after the job's terminal publish.
    """
    entered = threading.Event()
    running = threading.Event()
    port = _free_port()
    executor = _BlockingExecutor(entered, running)
    command = _dev_command()
    monkeypatch.setattr(_OneShotConsumer, "command", command, raising=False)

    agent = _make_agent(tmp_path, monkeypatch, port=port, executor=executor)
    # skip_self_update is a flag the REAL executor reads; this fake records the
    # call regardless, which is the point -- the boundary must be reached.
    run_task = asyncio.create_task(agent.run())
    try:
        await asyncio.to_thread(entered.wait, PHASE_ENTRY_TIMEOUT_SECONDS)
        agent._shutdown = True
        await asyncio.wait_for(run_task, timeout=90.0)
    finally:
        agent._shutdown = True

    assert executor.boundaries == [EnumSelfUpdateBoundary.POST_TERMINAL.value], (
        "the post-terminal self-update boundary did not fire through the "
        f"offloaded path: {executor.boundaries}"
    )
    job = agent.job_store.load(command.correlation_id)
    assert job is not None
    assert job.phase_results[Phase.PUBLISH] == PhaseStatus.SUCCESS, (
        "the boundary must come after the terminal publish, not instead of it"
    )


# --------------------------------------------------------------------------- #
# From the same diagnosis: name the phase that was RUNNING                     #
# --------------------------------------------------------------------------- #
def _crashed_job(phase_results: dict[Phase, PhaseStatus], current: Phase) -> JobState:
    return JobState(
        correlation_id=UUID(int=1),
        command={},
        current_phase=current,
        phase_results=phase_results,
        status="in_progress",
    )


@pytest.mark.unit
def test_an_interruption_names_the_phase_that_was_running() -> None:
    job = _crashed_job(
        {Phase.PREFLIGHT: PhaseStatus.SUCCESS, Phase.GIT: PhaseStatus.IN_PROGRESS},
        Phase.GIT,
    )
    assert describe_interruption(job) == f"interrupted during phase {Phase.GIT}"


@pytest.mark.unit
def test_an_interruption_between_phases_does_not_blame_a_completed_phase() -> None:
    """The RED case, and the reason this helper exists.

    ``update_phase`` writes ``current_phase`` on the SUCCESS update too, so a
    process killed after ``git`` succeeded and before ``compose_gen`` started
    left ``current_phase == git`` -- and the old string read "interrupted during
    phase git" about a phase that had completed, sending the reader to the wrong
    step. No phase is IN_PROGRESS in that state, and that is the fact the string
    must carry.
    """
    job = _crashed_job(
        {Phase.PREFLIGHT: PhaseStatus.SUCCESS, Phase.GIT: PhaseStatus.SUCCESS},
        Phase.GIT,
    )
    described = describe_interruption(job)
    assert described != f"interrupted during phase {Phase.GIT}", (
        "a phase that SUCCEEDED was named as the one that was interrupted"
    )
    assert "between phases" in described, described
    assert str(Phase.GIT) in described, (
        f"the last completed phase still bounds how far the job got: {described}"
    )


@pytest.mark.unit
def test_an_interruption_before_the_first_phase_says_so() -> None:
    job = _crashed_job({}, Phase.PREFLIGHT)
    assert describe_interruption(job) == "interrupted before any phase started"


@pytest.mark.unit
def test_recovery_records_which_phases_completed_and_what_was_running(
    tmp_path: Path,
) -> None:
    """The whole recovery record, on a job killed mid-phase.

    ``phase_results`` must still carry the completed phases -- an agent killed
    during a rebuild loses no evidence about what it had already done -- and the
    error string must name the phase that was running rather than the last one
    that started.
    """
    store = JobStore(state_dir=tmp_path)
    cid = uuid4()
    store.accept(cid, {})
    store.update_phase(cid, Phase.PREFLIGHT, PhaseStatus.IN_PROGRESS)
    store.update_phase(cid, Phase.PREFLIGHT, PhaseStatus.SUCCESS)
    store.update_phase(cid, Phase.GIT, PhaseStatus.IN_PROGRESS)
    store.update_phase(cid, Phase.GIT, PhaseStatus.SUCCESS)
    store.update_phase(cid, Phase.COMPOSE_GEN, PhaseStatus.IN_PROGRESS)

    recovered = store.recover_crashed_jobs()

    assert [job.correlation_id for job in recovered] == [cid]
    job = store.load(cid)
    assert job is not None
    assert job.status == "failed"
    assert job.phase_results[Phase.PREFLIGHT] == PhaseStatus.SUCCESS
    assert job.phase_results[Phase.GIT] == PhaseStatus.SUCCESS
    assert job.phase_results[Phase.COMPOSE_GEN] == PhaseStatus.FAILED
    assert job.phase_results[Phase.VERIFICATION] == PhaseStatus.SKIPPED
    assert job.errors == [f"interrupted during phase {Phase.COMPOSE_GEN}"], (
        f"the interruption must name the running phase: {job.errors}"
    )


@pytest.mark.unit
def test_recovery_between_phases_does_not_blame_the_completed_phase(
    tmp_path: Path,
) -> None:
    """The call site, not just the helper.

    RED against the tree where ``recover_crashed_jobs`` reads
    ``job.current_phase``: this job's last update was ``git: SUCCESS``, so
    ``current_phase`` reads ``git`` and the record claimed the git pull was
    interrupted. Nothing was running; the job died in the gap before
    ``compose_gen`` started, and sending a reader to the git phase costs them the
    time it takes to find nothing wrong there.
    """
    store = JobStore(state_dir=tmp_path)
    cid = uuid4()
    store.accept(cid, {})
    store.update_phase(cid, Phase.PREFLIGHT, PhaseStatus.IN_PROGRESS)
    store.update_phase(cid, Phase.PREFLIGHT, PhaseStatus.SUCCESS)
    store.update_phase(cid, Phase.GIT, PhaseStatus.IN_PROGRESS)
    store.update_phase(cid, Phase.GIT, PhaseStatus.SUCCESS)

    store.recover_crashed_jobs()

    job = store.load(cid)
    assert job is not None
    assert job.errors != [f"interrupted during phase {Phase.GIT}"], (
        "the recovery record blamed a phase that had SUCCEEDED; current_phase "
        "is written on the success update too and does not mean 'was running'"
    )
    assert len(job.errors) == 1
    assert "between phases" in job.errors[0], job.errors
    assert job.phase_results[Phase.GIT] == PhaseStatus.SUCCESS, (
        "a completed phase must stay completed on the recovery record"
    )
