# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Accept-backlog watchdog: an unserved listen queue must read RED (OMN-18636 AC4).

WHAT THIS EXISTS TO MAKE IMPOSSIBLE
-----------------------------------
A process whose event loop is blocked is not DOWN in any way a client can
observe. The kernel completes the TCP handshake on its behalf, queues the
connection, and accepts the request bytes into the socket buffer; the client
then waits for a reply that nothing is going to write. Measured 2026-09-17 on
the ``.201`` dev-lane agent (lane ``deploy-agent-http-hang-diag-2105``, 121
samples): 107 ``/health`` probes returned curl code ``000`` while
``ss -ltn 'sport = :8098'`` showed the accept queue climbing 9 -> 30 -> 68 -> 81
-> 129 against a backlog limit of 128, each queued connection separately visible
as ``CLOSE-WAIT`` holding 170 unread bytes of HTTP request.

To every reader downstream that is INDETERMINATE, not FAILED -- and
``INDETERMINATE`` is precisely the verdict that closed rule 24(b) delivery for a
sha whose lane had converged. AC4's wording is the design: "a watchdog that can
only report healthy-or-unreachable fails it, because unreachable is exactly the
state that reads as INDETERMINATE today."

WHY IT RUNS ON ITS OWN THREAD AND WRITES A FILE
-----------------------------------------------
Both halves follow from the condition being detected.

* **Its own thread**, because anything scheduled on the event loop is starved by
  exactly the state it is watching for. A loop-resident watchdog can only ever
  report healthy: when the answer would be "unhealthy", it does not get to run.
* **A durable record**, because when the loop is held the HTTP surface cannot
  carry the verdict either. The record is the readiness signal AC4 offers as the
  alternative to the health response -- a small JSON file beside the job records
  that a CI step can read with ``jq`` while the process is unreachable.

Both are also what the health response uses when it CAN answer: the same verdict
is embedded in ``/health``, which returns 503 while it is unhealthy, so the
surface goes red rather than silent.

WHAT THIS DELIBERATELY CANNOT DO (OMN-18636 AC6)
------------------------------------------------
It observes and reports. It holds no reference to the job pool, has no cancel,
no kill and no timeout, and nothing here can curtail an in-flight rebuild. A
deploy killed to keep a health surface green manufactures exactly the false FAIL
receipt this ticket exists to remove, so the fail-closed direction is: report
the truth, never act on it.

THE HONEST LIMIT
----------------
The depth is read from ``/proc/net/tcp``, which is Linux. Where it cannot be
read the verdict is INDETERMINATE -- never HEALTHY, because a blind watchdog
that reports healthy asserts the one thing it has no evidence for. An
indeterminate verdict does NOT turn the health response red: an unreadable
``/proc`` on a developer's laptop is not evidence of a saturated queue, and a
surface that is permanently red everywhere is a surface nobody reads. The agent
runs on Linux, where the probe is live.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

#: The file the watchdog writes beside the job records. Named, not derived, so a
#: workflow step can hardcode the same path this module writes.
ACCEPT_BACKLOG_RECORD_NAME = "accept_backlog.json"

#: How often the queue is sampled. One second is far below any bound worth
#: declaring and costs one small read of ``/proc/net/tcp`` per tick.
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0

#: THE DECLARED BOUND. How long the accept queue may stand non-empty with
#: nothing taken from it before the surface goes red.
#:
#: A queue that is briefly non-empty is what a listening socket does between the
#: handshake and the accept; that is not a finding at any depth. What is a
#: finding is a queue nothing has removed an entry from for this long, because
#: only ``accept()`` removes one -- the same reasoning the diagnosis used in the
#: other direction when a FALLING backlog proved the HTTP task was alive.
#:
#: Thirty seconds is fifteen times the 2 s bound the receipt reader allows per
#: request and an order of magnitude below the minutes a blocked loop held the
#: surface for. It is a bound on the DEFECT, not on a deploy: since the phases
#: moved off the loop (OMN-18636 AC1) a rebuild of any length leaves the queue
#: draining normally, so a trip here means the loop is blocked again, not that a
#: build is slow.
DEFAULT_UNDRAINED_BOUND_SECONDS = 30.0

_PROC_NET_TCP_FILES: tuple[str, ...] = ("/proc/net/tcp", "/proc/net/tcp6")

#: ``st`` column value for TCP_LISTEN in ``/proc/net/tcp``.
_TCP_LISTEN = "0A"


class EnumAcceptBacklogStatus(StrEnum):
    """Three answers, because two would force a lie.

    ``INDETERMINATE`` is not a shade of healthy. It is the state of a watchdog
    that could not read the queue, and it is reported as itself so that a reader
    can tell "the queue is fine" from "nobody looked".
    """

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    INDETERMINATE = "indeterminate"


class ModelAcceptQueueSample(BaseModel):
    """One observation of a listening socket's accept queue."""

    #: Connections the kernel has completed and the process has not accepted.
    depth: int


class ModelAcceptBacklogVerdict(BaseModel):
    """One watchdog verdict, as written to the readiness record and to /health.

    The field set is a contract with the CI reader, not an internal: it is read
    with ``jq`` from a workflow step while this process may be unreachable.
    """

    status: EnumAcceptBacklogStatus
    #: ``None`` only when the queue could not be read at all.
    queue_depth: int | None
    #: How long the queue has been continuously non-empty with no drain observed.
    undrained_seconds: float
    #: The bound this verdict was judged against, carried so a reader never has
    #: to know the default to interpret the number above.
    bound_seconds: float
    observed_at: datetime
    #: Why this verdict, in one sentence, naming the depth and the duration.
    evidence: str


def read_accept_queue(port: int) -> ModelAcceptQueueSample | None:
    """Read the accept-queue depth of the listening socket on ``port``.

    The kernel reports a listening socket's ``sk_ack_backlog`` -- the count of
    established connections waiting for ``accept()`` -- in the ``rx_queue`` half
    of the ``tx_queue:rx_queue`` column of ``/proc/net/tcp``. That is the same
    number ``ss -ltn`` prints as ``Recv-Q``, which is the number the diagnosis
    measured climbing to 129.

    The backlog LIMIT is deliberately not reported. ``ss`` reads it from the
    netlink sock-diag interface, not from ``/proc``, where a listening socket's
    ``tx_queue`` column is always zero; returning that zero as a limit would be
    a fabricated fact in a record whose whole purpose is evidence.

    Returns ``None`` -- never a zero -- when the queue cannot be read, whether
    because the host has no ``/proc/net/tcp`` or because no socket is listening
    on that port. A zero means "read it, and it is empty"; the two must not
    collapse, because one of them is healthy and the other is unknown.
    """
    best: int | None = None
    for path in _PROC_NET_TCP_FILES:
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 5 or fields[3] != _TCP_LISTEN:
                continue
            local = fields[1]
            _, _, local_port_hex = local.rpartition(":")
            try:
                if int(local_port_hex, 16) != port:
                    continue
                depth = int(fields[4].split(":")[1], 16)
            except (ValueError, IndexError):
                continue
            # The maximum across listeners, not the sum: a dual-stack socket
            # appears once, and two sockets on one port are two listeners of
            # which the worst-off is the one worth reporting.
            best = depth if best is None else max(best, depth)
    if best is None:
        return None
    return ModelAcceptQueueSample(depth=best)


def read_verdict_record(state_dir: Path | str) -> ModelAcceptBacklogVerdict | None:
    """Read the readiness record, treating unreadable as absent.

    The consumer is CI code reading a file another process writes once a second,
    so a truncated read is an ordinary event rather than an error condition.
    Absent, corrupt and schema-invalid all mean the same thing to a reader --
    "no usable verdict" -- and none of them is a healthy one. This never raises,
    because an exception out of a readiness read turns a missing signal into a
    failed job for an unrelated reason.
    """
    path = Path(state_dir) / ACCEPT_BACKLOG_RECORD_NAME
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ModelAcceptBacklogVerdict.model_validate_json(payload)
    except ValueError:
        return None


class AcceptBacklogWatchdog:
    """Samples the listen socket's accept queue from a thread of its own.

    See the module docstring for why the thread and the file are both load
    bearing. The decision rule is in ``observe``, which is pure and takes the
    sample as an argument, so the rule is asserted directly rather than through
    a sleep race.
    """

    def __init__(
        self,
        *,
        port: int,
        state_dir: Path | str,
        bound_seconds: float = DEFAULT_UNDRAINED_BOUND_SECONDS,
        interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        probe: Callable[[], ModelAcceptQueueSample | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._port = port
        self._state_dir = Path(state_dir)
        self._bound_seconds = float(bound_seconds)
        self._interval_seconds = float(interval_seconds)
        self._probe = probe if probe is not None else (lambda: read_accept_queue(port))
        self._clock = clock
        self._wall_clock = wall_clock
        # State of the current undrained run: when it started, and the shallowest
        # depth seen in it. A sample below that minimum is proof `accept()` ran.
        self._run_started_at: float | None = None
        self._run_min_depth: int | None = None
        self._latest: ModelAcceptBacklogVerdict | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def record_path(self) -> Path:
        return self._state_dir / ACCEPT_BACKLOG_RECORD_NAME

    def latest(self) -> ModelAcceptBacklogVerdict | None:
        with self._lock:
            return self._latest

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def observe(
        self, sample: ModelAcceptQueueSample | None
    ) -> ModelAcceptBacklogVerdict:
        """Fold one sample into the run state and return the verdict.

        The rule, in the terms the diagnosis established:

        * **no sample** -- INDETERMINATE, and the run is forgotten. A gap in
          observation is not evidence that the queue stood still through it.
        * **depth 0** -- the queue is empty, so nothing is waiting: HEALTHY, run
          cleared.
        * **depth below the run's minimum** -- an entry was removed, and only
          ``accept()`` removes one, so the loop is alive. The run RESTARTS from
          here rather than continuing, because the elapsed time before a proven
          accept says nothing about the time after it.
        * **otherwise** -- the run continues, and once it has stood longer than
          the declared bound the verdict is UNHEALTHY.
        """
        now = self._clock()
        observed_at = self._wall_clock()

        if sample is None:
            self._run_started_at = None
            self._run_min_depth = None
            return self._record(
                ModelAcceptBacklogVerdict(
                    status=EnumAcceptBacklogStatus.INDETERMINATE,
                    queue_depth=None,
                    undrained_seconds=0.0,
                    bound_seconds=self._bound_seconds,
                    observed_at=observed_at,
                    evidence=(
                        f"the accept queue for port {self._port} could not be "
                        "read (no /proc/net/tcp entry for a listening socket on "
                        "that port), so this process cannot tell a healthy "
                        "queue from a saturated one"
                    ),
                )
            )

        depth = sample.depth
        if depth == 0:
            self._run_started_at = None
            self._run_min_depth = None
            return self._record(
                ModelAcceptBacklogVerdict(
                    status=EnumAcceptBacklogStatus.HEALTHY,
                    queue_depth=0,
                    undrained_seconds=0.0,
                    bound_seconds=self._bound_seconds,
                    observed_at=observed_at,
                    evidence=(
                        f"the accept queue for port {self._port} is empty: "
                        "every completed connection has been accepted"
                    ),
                )
            )

        drained = self._run_min_depth is not None and depth < self._run_min_depth
        if self._run_started_at is None or drained:
            self._run_started_at = now
            self._run_min_depth = depth
        else:
            self._run_min_depth = min(self._run_min_depth or depth, depth)

        undrained = max(0.0, now - self._run_started_at)
        if undrained >= self._bound_seconds and not drained:
            status = EnumAcceptBacklogStatus.UNHEALTHY
            evidence = (
                f"{depth} connection(s) have been waiting on port {self._port}'s "
                f"accept queue for {undrained:.0f}s with no accept observed, "
                f"past the declared bound of {self._bound_seconds:.0f}s: the "
                "event loop is not serving this socket"
            )
        else:
            status = EnumAcceptBacklogStatus.HEALTHY
            evidence = (
                f"{depth} connection(s) waiting on port {self._port}'s accept "
                f"queue for {undrained:.0f}s, within the declared bound of "
                f"{self._bound_seconds:.0f}s"
            )
        return self._record(
            ModelAcceptBacklogVerdict(
                status=status,
                queue_depth=depth,
                undrained_seconds=undrained,
                bound_seconds=self._bound_seconds,
                observed_at=observed_at,
                evidence=evidence,
            )
        )

    def sample_once(self) -> ModelAcceptBacklogVerdict:
        """Probe, decide, persist. The unit of work the thread repeats."""
        try:
            sample = self._probe()
        except Exception:
            logger.exception("accept-backlog probe raised; reporting indeterminate")
            sample = None
        return self.observe(sample)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="deploy-agent-accept-backlog", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception:
                # The watchdog must not be the thing that takes the agent down.
                logger.exception("accept-backlog watchdog sample failed")
            self._stop.wait(self._interval_seconds)

    def _record(self, verdict: ModelAcceptBacklogVerdict) -> ModelAcceptBacklogVerdict:
        with self._lock:
            self._latest = verdict
        self._write(verdict)
        return verdict

    def _write(self, verdict: ModelAcceptBacklogVerdict) -> None:
        """Write the readiness record atomically.

        Atomic because the reader is another process polling it: a partial file
        would read as corrupt, and corrupt reads as absent, so a torn write would
        silently erase the signal at exactly the moment it matters.
        """
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._state_dir, suffix=".tmp")
            try:
                os.write(fd, verdict.model_dump_json(indent=2).encode())
            finally:
                os.close(fd)
            Path(tmp).replace(self.record_path)
        except OSError:
            logger.exception(
                "accept-backlog verdict could not be written to %s; the CI "
                "reader will see no readiness signal for this agent",
                self.record_path,
            )
