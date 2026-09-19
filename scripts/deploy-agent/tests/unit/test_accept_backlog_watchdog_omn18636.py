# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18636 AC4 -- an unserved accept backlog must read RED, never silent.

THE DEFECT THESE TESTS PIN
--------------------------
The kernel completes the TCP handshake for a listening socket whether or not the
process ever calls ``accept()``. So an agent whose event loop is blocked is not
DOWN in any way a client can see: the connection is established, the request
bytes are delivered into the socket buffer, and the client waits. Measured
2026-09-17 (lane ``deploy-agent-http-hang-diag-2105``): 107 of 121 ``/health``
probes returned curl code ``000`` while ``ss -ltn 'sport = :8098'`` showed
Recv-Q climbing 9 -> 30 -> 68 -> 81 -> 129 against a backlog limit of 128, each
queued connection separately visible as ``CLOSE-WAIT`` with 170 unread bytes.

Downstream that reads as ``INDETERMINATE``, not as a failure -- and
``INDETERMINATE`` is exactly the verdict this ticket exists to stop producing
for a lane that had in fact converged.

WHY THE WATCHDOG RUNS ON ITS OWN THREAD
---------------------------------------
This is the whole design, and the criterion names it: "a watchdog that can only
report healthy-or-unreachable fails it". A watchdog scheduled on the event loop
is starved by precisely the condition it exists to detect, so it can only ever
report healthy. This one samples from an OS thread and writes a durable record,
so its verdict is produced and readable while the loop is held -- and
``test_the_verdict_is_produced_while_the_loop_is_held`` holds the loop and reads
the record to prove it, rather than reading the source for a ``Thread``.

THE THREE-VALUED STATUS, AND WHY IT IS NOT TWO
----------------------------------------------
``/proc/net/tcp`` is Linux. On a host where the queue cannot be read, the honest
answer is INDETERMINATE -- "this process does not know" -- and never HEALTHY. A
watchdog that reported healthy when blind would assert the one thing it has no
evidence for. It does not turn the health response red either: an unreadable
``/proc`` on a developer's machine is not evidence of a saturated queue, and a
surface that is permanently red on every non-Linux host is a surface nobody
reads. The deploy agent runs on Linux, where the probe is live.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.accept_backlog import (
    ACCEPT_BACKLOG_RECORD_NAME,
    AcceptBacklogWatchdog,
    EnumAcceptBacklogStatus,
    ModelAcceptQueueSample,
    read_accept_queue,
    read_verdict_record,
)
from deploy_agent.agent import DeployAgent

pytestmark = pytest.mark.unit

SHA = "c" * 40


class _Clock:
    """A monotonic clock the test advances by hand.

    The decision under test is "non-empty and undrained for longer than a
    declared bound". Driving it from real time would make the test a sleep race;
    driving it from a handle makes the boundary exact, so the assertion is about
    the rule rather than about the machine the suite is running on.
    """

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _watchdog(
    tmp_path: Path,
    *,
    clock: _Clock,
    bound: float = 30.0,
    probe: Any = None,
) -> AcceptBacklogWatchdog:
    return AcceptBacklogWatchdog(
        port=8098,
        state_dir=tmp_path,
        bound_seconds=bound,
        interval_seconds=0.01,
        probe=probe or (lambda: None),
        clock=clock,
    )


# --------------------------------------------------------------------------- #
# The decision rule                                                            #
# --------------------------------------------------------------------------- #
def test_an_undrained_queue_turns_unhealthy_only_after_the_declared_bound(
    tmp_path: Path,
) -> None:
    """The bound is a bound: before it healthy, at it unhealthy, with evidence.

    A queue that is momentarily non-empty is normal -- it is what a socket does
    between the handshake and the accept. What is not normal is a queue nothing
    has taken from for longer than the declared bound, which is the signature of
    a loop that is not calling ``accept()`` at all.
    """
    clock = _Clock()
    watchdog = _watchdog(tmp_path, clock=clock, bound=30.0)

    first = watchdog.observe(ModelAcceptQueueSample(depth=4))
    assert first.status is EnumAcceptBacklogStatus.HEALTHY, (
        f"a queue that has only just become non-empty is not a finding: {first}"
    )

    clock.advance(29.0)
    still_inside = watchdog.observe(ModelAcceptQueueSample(depth=9))
    assert still_inside.status is EnumAcceptBacklogStatus.HEALTHY, (
        f"the bound was not yet exceeded: {still_inside}"
    )

    clock.advance(2.0)
    verdict = watchdog.observe(ModelAcceptQueueSample(depth=17))

    assert verdict.status is EnumAcceptBacklogStatus.UNHEALTHY, (
        "a queue that has been non-empty and undrained past the bound is the "
        f"blocked-loop signature and must read RED: {verdict}"
    )
    assert verdict.queue_depth == 17, (
        f"the criterion requires the OBSERVED depth in the evidence: {verdict}"
    )
    assert verdict.undrained_seconds >= 30.0
    assert "17" in verdict.evidence and "31" in verdict.evidence, (
        "the evidence string must carry the depth and how long it has stood, "
        f"because that pair is what distinguishes this from a busy moment: "
        f"{verdict.evidence}"
    )


def test_a_falling_depth_proves_accept_ran_and_resets_the_run(
    tmp_path: Path,
) -> None:
    """Only ``accept()`` removes an entry, so a fall is proof of life.

    This is the reasoning the diagnosis used to rule out a dead HTTP task: the
    backlog FELL from 32 to 9 between two measurements, which no dead task can
    produce. The watchdog uses the same fact in the same direction -- a drain
    clears the run, so a busy agent that is nonetheless accepting never trips.
    """
    clock = _Clock()
    watchdog = _watchdog(tmp_path, clock=clock, bound=30.0)

    watchdog.observe(ModelAcceptQueueSample(depth=32))
    clock.advance(25.0)
    watchdog.observe(ModelAcceptQueueSample(depth=32))

    clock.advance(1.0)
    drained = watchdog.observe(ModelAcceptQueueSample(depth=9))
    assert drained.status is EnumAcceptBacklogStatus.HEALTHY
    assert drained.undrained_seconds == 0.0, (
        f"a drain restarts the run rather than shortening it: {drained}"
    )

    # And the clock that had nearly run out must genuinely start over.
    clock.advance(25.0)
    after = watchdog.observe(ModelAcceptQueueSample(depth=11))
    assert after.status is EnumAcceptBacklogStatus.HEALTHY, (
        "the pre-drain elapsed time leaked into the new run, so an agent that "
        f"is accepting normally would be reported unhealthy: {after}"
    )


def test_an_empty_queue_is_healthy_and_clears_a_standing_run(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    watchdog = _watchdog(tmp_path, clock=clock, bound=1.0)
    watchdog.observe(ModelAcceptQueueSample(depth=5))
    clock.advance(10.0)
    assert watchdog.observe(ModelAcceptQueueSample(depth=5)).status is (
        EnumAcceptBacklogStatus.UNHEALTHY
    )

    recovered = watchdog.observe(ModelAcceptQueueSample(depth=0))
    assert recovered.status is EnumAcceptBacklogStatus.HEALTHY
    assert recovered.queue_depth == 0


def test_an_unreadable_queue_is_indeterminate_and_never_healthy(
    tmp_path: Path,
) -> None:
    """Blind is a third answer. Reporting healthy while blind is the lie."""
    clock = _Clock()
    watchdog = _watchdog(tmp_path, clock=clock)

    verdict = watchdog.observe(None)

    assert verdict.status is EnumAcceptBacklogStatus.INDETERMINATE, (
        f"an unreadable accept queue must not read as healthy: {verdict}"
    )
    assert verdict.queue_depth is None
    assert verdict.evidence, "an indeterminate verdict must say why it is blind"


# --------------------------------------------------------------------------- #
# The probe really reads the kernel's accept queue                             #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason=(
        "the accept-queue depth is read from /proc/net/tcp, which is Linux. The "
        "deploy agent runs on Linux (.201 and the runner fleet); this test is "
        "what proves the probe observes the kernel rather than a stub, so it "
        "runs there and is skipped rather than faked elsewhere."
    ),
)
def test_read_accept_queue_observes_connections_nothing_has_accepted() -> None:
    """Connect without accepting, and the depth must move.

    A probe that returned a plausible number without reading the kernel would
    pass every other test in this file. This one binds a real listening socket,
    opens real connections and never calls ``accept()`` -- reproducing in
    miniature exactly what the blocked agent did to 129 CI connections.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clients: list[socket.socket] = []
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = int(listener.getsockname()[1])

        empty = read_accept_queue(port)
        assert empty is not None, "a listening socket was not found in /proc/net/tcp"
        assert empty.depth == 0

        for _ in range(3):
            client = socket.create_connection(("127.0.0.1", port), timeout=5)
            clients.append(client)
        # The handshake completes in the kernel, asynchronously to this thread.
        deadline = time.monotonic() + 5.0
        sample = read_accept_queue(port)
        while (sample is None or sample.depth < 3) and time.monotonic() < deadline:
            time.sleep(0.05)
            sample = read_accept_queue(port)

        assert sample is not None
        assert sample.depth == 3, (
            "the probe did not observe three established-but-unaccepted "
            f"connections: {sample}"
        )
        # The backlog LIMIT is deliberately not reported: /proc gives a
        # listening socket a zero tx_queue, and ss reads the real limit over
        # netlink. A fabricated 0 in an evidence record is worse than no field.
        assert not hasattr(sample, "limit"), (
            "a limit read from /proc/net/tcp would be a fabricated zero"
        )
    finally:
        for client in clients:
            client.close()
        listener.close()


# --------------------------------------------------------------------------- #
# AC4's falsifier: the verdict exists while the loop is held                    #
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _IdleExecutor:
    def __init__(self) -> None:
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

    def self_update(self, *, boundary: Any, **kwargs: Any) -> None:
        pass


class _SilentConsumer:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def poll_and_accept(self) -> tuple[None, None]:
        time.sleep(0.05)
        return None, None

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_the_verdict_is_produced_while_the_loop_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold the loop for longer than the bound; the watchdog must still speak.

    This is AC4's falsifier. The loop is blocked here on purpose -- a
    ``time.sleep`` inside the test's own coroutine holds the exact thread a
    regression would hold -- and the assertion is made against the DURABLE
    RECORD, read from the test thread. A watchdog living on the loop cannot
    write that record while the loop is held, which is what makes this a
    falsifier rather than a formality: it fails against the healthy-or-silent
    design the criterion rejects, and passes only against one that runs off the
    loop and leaves its answer somewhere a reader can find it.

    The probe is injected so the depth is the test's to decide. What the probe
    reads from the kernel is proven separately, above, on Linux; conflating the
    two would make this test unrunnable on a developer's machine and prove
    neither thing well.
    """
    port = _free_port()
    bound = 0.4
    depth = 37

    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy.lock")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "jobs")
    monkeypatch.setattr(agent_mod, "HEALTH_PORT", port)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    monkeypatch.setattr(agent_mod, "DeployConsumer", _SilentConsumer)
    monkeypatch.setattr(agent_mod, "record_loaded_code_sha", lambda _agent_dir: SHA)
    monkeypatch.setattr(agent_mod, "ACCEPT_BACKLOG_BOUND_SECONDS", bound)
    monkeypatch.setattr(agent_mod, "ACCEPT_BACKLOG_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(
        agent_mod,
        "read_accept_queue",
        lambda _port: ModelAcceptQueueSample(depth=depth),
    )

    agent = DeployAgent(skip_self_update=True)
    agent.executor = _IdleExecutor()  # type: ignore[assignment]

    run_task = asyncio.create_task(agent.run())
    try:
        # Let the agent bind and the watchdog take its first samples.
        await asyncio.sleep(0.3)
        # HOLD THE LOOP. Nothing scheduled on it runs for this whole span.
        time.sleep(bound * 4)

        record = read_verdict_record(tmp_path / "jobs")
        assert record is not None, (
            "no verdict was written while the loop was held, so the watchdog "
            "is on the loop it is meant to be watching -- the design AC4 "
            f"rejects. expected {tmp_path / 'jobs' / ACCEPT_BACKLOG_RECORD_NAME}"
        )
        assert record.status is EnumAcceptBacklogStatus.UNHEALTHY, (
            "the accept queue stood at depth "
            f"{depth} for longer than the {bound}s bound while nothing accepted "
            f"from it, and the watchdog still reported: {record}"
        )
        assert record.queue_depth == depth, (
            f"the criterion requires the observed depth in the evidence: {record}"
        )
        assert str(depth) in record.evidence
        assert record.observed_at.tzinfo is not None, (
            "a readiness record a CI reader consumes must carry an unambiguous "
            f"timestamp, or it cannot be checked for staleness: {record}"
        )

        # And the surface itself must go red rather than stay quiet.
        status, body = await asyncio.to_thread(
            _get_health, f"http://127.0.0.1:{port}/health"
        )
        assert status == 503, (
            "the health response stayed green while the agent's own watchdog "
            f"held an unhealthy verdict: status={status} body={body}"
        )
        assert body is not None
        assert body["accept_backlog"]["status"] == (
            EnumAcceptBacklogStatus.UNHEALTHY.value
        )
        assert body["accept_backlog"]["queue_depth"] == depth
    finally:
        agent._shutdown = True
        await asyncio.wait_for(run_task, timeout=60.0)


def _get_health(url: str) -> tuple[int | None, dict[str, Any] | None]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5.0) as response:
            return int(response.status), json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # A 503 is the expected answer here, and urllib raises on it.
        return int(exc.code), json.loads(exc.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None, None


def test_the_watchdog_stops_and_leaves_its_last_verdict_readable(
    tmp_path: Path,
) -> None:
    """Stopping is clean, and the record survives it.

    A record that is deleted on shutdown cannot answer the question a reader
    asks after the fact, and a thread that does not join turns every test run
    after this one into a race.
    """
    clock = _Clock()
    watchdog = AcceptBacklogWatchdog(
        port=8098,
        state_dir=tmp_path,
        bound_seconds=0.1,
        interval_seconds=0.01,
        probe=lambda: ModelAcceptQueueSample(depth=2),
        clock=clock,
    )
    watchdog.start()
    try:
        deadline = time.monotonic() + 5.0
        while watchdog.latest() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert watchdog.latest() is not None, "the watchdog thread never sampled"
    finally:
        watchdog.stop()

    assert not watchdog.is_running(), "the watchdog thread did not stop"
    record = read_verdict_record(tmp_path)
    assert record is not None, "the last verdict did not survive shutdown"
    assert record.queue_depth == 2


def test_a_corrupt_record_reads_as_absent_rather_than_raising(
    tmp_path: Path,
) -> None:
    """The reader is CI code; a half-written file must not crash it.

    Absent and corrupt are both "no usable verdict" to a consumer, and neither
    is a healthy one -- what must not happen is an exception out of a readiness
    read, because that turns a missing signal into a failed job for an unrelated
    reason.
    """
    (tmp_path / ACCEPT_BACKLOG_RECORD_NAME).write_text("{not json", encoding="utf-8")
    assert read_verdict_record(tmp_path) is None
    assert read_verdict_record(tmp_path / "nonexistent") is None


def test_the_record_is_json_a_shell_reader_can_parse(tmp_path: Path) -> None:
    """The record is consumed by CI, so its shape is a contract, not an internal.

    Named fields, an ISO timestamp and a status string -- readable with ``jq``
    from a workflow step without importing this package.
    """
    clock = _Clock()
    watchdog = _watchdog(
        tmp_path,
        clock=clock,
        bound=0.0,
        probe=lambda: ModelAcceptQueueSample(depth=5),
    )
    watchdog.sample_once()

    payload = json.loads((tmp_path / ACCEPT_BACKLOG_RECORD_NAME).read_text())
    assert set(payload) >= {
        "status",
        "queue_depth",
        "undrained_seconds",
        "bound_seconds",
        "observed_at",
        "evidence",
    }
    assert payload["status"] in {status.value for status in EnumAcceptBacklogStatus}
    assert datetime.fromisoformat(payload["observed_at"]).tzinfo is not None
    assert datetime.fromisoformat(payload["observed_at"]) <= datetime.now(UTC)


def test_the_watchdog_never_touches_the_job_it_is_watching(tmp_path: Path) -> None:
    """AC6, from this file's side: the watchdog reports and does nothing else.

    The failure mode a health fix invites is keeping the surface responsive by
    curtailing the work behind it. This watchdog has no handle on the job pool,
    no cancel, no kill and no timeout -- it samples a socket and writes a file.
    The absence is asserted here because the temptation to add one lands exactly
    here, on the class that has the evidence a kill would seem justified by.
    """
    watchdog = AcceptBacklogWatchdog(
        port=8098,
        state_dir=tmp_path,
        probe=lambda: ModelAcceptQueueSample(depth=200),
    )
    forbidden = {"cancel", "kill", "terminate", "abort", "shutdown_agent", "interrupt"}
    assert not forbidden & set(dir(watchdog)), (
        "the watchdog grew a way to curtail the work it observes, which is the "
        "AC6 fail-closed direction: a deploy killed to keep a surface green "
        "manufactures the false FAIL receipt this ticket removes"
    )
    # And it is constructible without any reference to the agent or its pool.
    assert not any(
        isinstance(getattr(watchdog, name, None), threading.Thread)
        for name in ("_job_pool", "executor")
    )
