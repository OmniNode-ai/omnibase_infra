# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Never cancel a deps-phase compose recreate mid-removal (OMN-18692).

WHY THIS EXISTS
---------------

MEASURED on the ``.201`` dev lane, 2026-09-18 (``docs/tracking/
ROLLING_WORK_LEDGER.md:4256``). Job ``6cee06d3-7337-4263-a12b-851a46e0bf01``
(scope=full) was accepted at 12:55:04Z under ``load1 28.42`` on 32 cores. Its
CORE phase ran ``docker compose up -d --force-recreate`` against the flat
``PHASE_TIMEOUTS[Phase.CORE] = 300``. The agent's own log, verbatim at
13:13:29Z::

    docker compose up exceeded its 300s ceiling for phase core and was killed
    with the lane mid-recreate

dockerd logged ``Canceled: context canceled`` against four
``redpandadata/redpanda:v24.2.7`` containers at 13:01:56Z. The lane was left
with **redpanda removed and never recreated**, valkey removed and postgres
half-recreated as ``7a423d93d83e_omnibase-infra-postgres``. The agent's own
control bus IS that broker -- ``deploy-agent-dev.service`` points its
``KAFKA_BOOTSTRAP_SERVERS`` at the dev lane's own external Redpanda listener --
so it then crash-looped on ``NoBrokersAvailable`` against the broker it had
just destroyed and systemd gave up at 13:17:31Z with restart counter 7. The
fleet's CI bus is that same listener, so every OCC autobind across four
repositories failed closed for about 35 minutes.

OMN-18057 fixed the RUNTIME phase of this shape and is NOT regressed: the
runtime compose-up ceiling is still derived from the compose model
(``compose_budget``) and a blown runtime ceiling still routes into recovery.
What it did not cover is the CORE phase, whose entry in ``PHASE_TIMEOUTS``
stayed a bare ``300`` because the runtime ceiling was the one that had killed a
lane. This module is the deps-phase half, and it is deliberately a DIFFERENT
mechanism rather than a second ceiling, because the two failures are not the
same failure:

* the runtime kill left containers in ``Created`` -- bad, recoverable in place;
* the deps kill left containers **absent**, including the one the agent reads
  its own commands from -- so the process that would have recovered them died.

THE ASYMMETRY IS THE WHOLE DESIGN. Over-waiting on a deps recreate costs a slow
deploy that the agent's HTTP surface reports on throughout (OMN-18636 landed
that surface's responsiveness during a phase). Under-waiting destroys the lane,
the control bus and the fleet's merge gate at once. So every degraded reading
here WIDENS, exactly as ``host_conditions`` records for the build ceiling, and
the one thing that is refused outright is starting a removal this module cannot
promise to finish.

THREE MECHANISMS, IN THE ORDER THEY APPLY
-----------------------------------------

1. **Defer before touching anything.** ``defer_until_host_quiesces`` reads
   ``/proc/loadavg`` through ``host_conditions`` and, above a committed
   saturation threshold, WAITS rather than starting a removal. Nothing has been
   stopped, created or removed while it waits, so a deferral that ends in a
   refusal leaves the lane byte-identical to how it found it. The refusal is a
   named outcome the receipt carries, not a bare timeout.

2. **A ceiling anchored at the first container start, scaled by host load.**
   ``derive_deps_recreate_budget`` takes the compose model's own number (the
   largest health-gated ``start_period`` among the deps, read by
   ``compose_budget``) and multiplies it by a contention term. The supervisor
   then starts the clock at the first OBSERVED change to the lane's container
   state, not at command start, because the seconds before compose touches
   anything are queueing on a loaded host and spending the recreate's budget on
   them is what made a 300s ceiling fire during a removal.

3. **A kill is refused while the lane is mid-recreate and still moving.** Past
   the ceiling the supervisor keeps waiting for as long as the lane's container
   state keeps CHANGING. Only a genuinely stalled lane, or the hard upper
   bound, ends the wait -- and both of those endings are named separately in
   the receipt so "we killed a hung recreate" is never recorded as "we killed a
   slow one".

WHAT IS DELIBERATELY NOT CLAIMED
--------------------------------

The hard upper bound still ends in a kill. It is not a cure for a genuinely
wedged dockerd, and nothing here can make a removal atomic -- compose removes
before it creates and this module does not change that. What it removes is the
case measured above: a healthy recreate killed by a clock that could not see
the machine it was running on. The caller is expected to converge the deps
after ANY killed outcome; ``DeployExecutor._compose_up`` does, and
``DeployAgent`` does the same at startup so a crashed process cannot consume a
new command against an absent broker.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from deploy_agent.compose_budget import ModelPhaseBudget
from deploy_agent.events import EnumRecreateOutcome, ModelRecreateSupervision
from deploy_agent.host_conditions import ModelHostConditions, probe_host_conditions

logger = logging.getLogger(__name__)

#: Above this saturation (``load1 / cpu_count``) a deps recreate starts
#: competing for CPU the ceiling implicitly assumed it had. DELIBERATELY LOWER
#: than ``host_conditions.CONTENTION_SATURATION_THRESHOLD`` (1.0), which is
#: calibrated for a BuildKit solve. The measured kill happened at saturation
#: 28.42/32 = 0.888, BELOW 1.0, so a term that only engages at 1.0 would have
#: left this incident's ceiling exactly where it was.
DEPS_CONTENTION_SATURATION_THRESHOLD = 0.5

#: How much of the excess saturation is charged to the ceiling. UNDAMPED, at
#: 2.0, and that is the opposite calibration to the build term's 0.5 for a
#: stated reason: a container stop/create/start is dominated by the daemon and
#: the kernel doing work this process is queueing behind, not by cache-served
#: layers, so contention costs it closer to linearly than it costs a solve.
DEPS_CONTENTION_SLOPE = 2.0

#: The widest the contention term alone may open the deps ceiling.
MAX_DEPS_CONTENTION_MULTIPLIER = 4.0

#: An unreadable machine is charged the worst case it might be hiding, because
#: the direction of every degraded reading here is outward. See the module
#: docstring for why that is not a fail-open.
UNKNOWN_DEPS_CONTENTION_MULTIPLIER = MAX_DEPS_CONTENTION_MULTIPLIER

#: Added to the largest health-gated ``start_period`` among the deps, the same
#: shape ``RUNTIME_COMPOSE_UP_MARGIN_SECONDS`` already uses for the runtime
#: phase: it covers the unhealthy-detection tail the compose file declares plus
#: the stop/create/start of the rest of the selected set.
DEPS_COMPOSE_UP_MARGIN_SECONDS = 300

#: The floor the derived deps ceiling can never fall below, BEFORE the host
#: term. MEASURED both ways: the recovery lane's own
#: ``up -d postgres redpanda valkey`` on a quiet host had all three Up healthy
#: in 32s (13:35:56Z -> 13:36:28Z), while the same work under saturation 0.888
#: outran 300s and was killed -- a RIGHT-CENSORED observation, so its true
#: duration is unknown and is only known to exceed 300s. A floor is therefore
#: set from the failure's blast radius rather than from a measured success:
#: 900s is three times the ceiling that destroyed the lane, and being wrong
#: upward here costs a slow deploy that ``/health`` reports on throughout.
DEPS_COMPOSE_UP_FLOOR_SECONDS = 900

#: What keeps "adapts to the host" from meaning "unbounded". No combination of
#: derived term, margin, floor and host multiplier may exceed this, and the
#: supervisor stops waiting here even on a lane that is still changing. The
#: same safety property ``build_budget.HARD_UPPER_BOUND_SECONDS`` holds for the
#: image build.
DEPS_COMPOSE_UP_HARD_UPPER_BOUND_SECONDS = 3600

#: Past the ceiling, the supervisor keeps waiting while the lane's container
#: state keeps changing. This is how long it tolerates NO change at all before
#: reading the recreate as wedged rather than slow. A stopped container that
#: never gets recreated changes nothing, which is exactly the state a stalled
#: removal sits in.
MID_RECREATE_STALL_SECONDS = 600

#: How often the supervisor samples the lane while compose runs. Cheap
#: (``docker compose ps``) and far below every bound above, so the anchor and
#: the stall detector both have resolution to work with.
SUPERVISOR_POLL_INTERVAL_SECONDS = 5

#: How long a terminated compose CLI is given to unwind before SIGKILL.
TERMINATE_GRACE_SECONDS = 30

#: The committed threshold above which a deps recreate is NOT STARTED. Higher
#: than the ceiling's engagement threshold on purpose: the ceiling widening is
#: the mechanism that covers an ordinarily busy host, and this is the second
#: line for a host so loaded that no ceiling can be promised. At 1.5 the
#: measured incident (0.888) would NOT have deferred -- correctly, because what
#: that incident needed was the wider ceiling, not a refusal.
DEPS_RECREATE_DEFER_SATURATION = 1.5

#: How long the deferral waits for the host to quiesce before refusing. Nothing
#: is mutated during this wait.
DEPS_RECREATE_DEFER_MAX_WAIT_SECONDS = 900

#: How often the deferral re-reads the machine while it waits.
DEPS_RECREATE_DEFER_POLL_SECONDS = 30

StateReader = Callable[[], Mapping[str, tuple[str, int | None]]]
HostProbe = Callable[[], ModelHostConditions]
#: Returns a monotonically non-decreasing count of bytes the compose CLI has
#: written. See ``supervise_compose_up`` for why the lane's container states
#: are not a sufficient liveness signal on their own.
ProgressReader = Callable[[], int]


class HostContentionDeferralError(RuntimeError):
    """The host never quiesced, so the recreate was refused BEFORE any removal.

    Its own class, not a generic ``RuntimeError``, because the fact that
    distinguishes it from every other deploy failure is that THE LANE WAS NOT
    TOUCHED. A caller reading this knows no container was stopped, created or
    removed, which is the difference between "retry it" and "go and look at the
    lane first".
    """


class ModelDepsRecreateBudget(BaseModel):
    """A deps-phase compose-up ceiling and the facts it was derived from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_seconds: int
    hard_upper_bound_seconds: int
    model_seconds: int
    host_multiplier: float
    saturation: float | None
    source_service: str | None
    clamped_to_hard_upper_bound: bool

    def describe(self) -> str:
        """One-line, log-ready statement of the ceiling and its two sources."""
        source = (
            f"start_period of {self.source_service!r}"
            if self.source_service
            else "no health-gated start_period in the compose model; floor"
        )
        load = (
            "load1 unreadable"
            if self.saturation is None
            else f"saturation {self.saturation:.2f}"
        )
        clamp = (
            f", CLAMPED to the {self.hard_upper_bound_seconds}s hard upper bound"
            if self.clamped_to_hard_upper_bound
            else ""
        )
        return (
            f"{self.timeout_seconds}s = compose model {self.model_seconds}s "
            f"({source}) x host {self.host_multiplier:.2f} ({load}){clamp}"
        )


def deps_contention_multiplier(saturation: float | None) -> float:
    """Return the widening factor a machine at ``saturation`` earns."""
    if saturation is None:
        return UNKNOWN_DEPS_CONTENTION_MULTIPLIER
    excess = saturation - DEPS_CONTENTION_SATURATION_THRESHOLD
    if excess <= 0:
        return 1.0
    return min(MAX_DEPS_CONTENTION_MULTIPLIER, 1.0 + excess * DEPS_CONTENTION_SLOPE)


def derive_deps_recreate_budget(
    model_budget: ModelPhaseBudget,
    host: ModelHostConditions,
) -> ModelDepsRecreateBudget:
    """Combine the compose model's own number with the machine it runs on.

    ``model_budget`` comes from ``compose_budget.derive_runtime_phase_budget``
    over the DEPS service list, so a compose change that moves a deps
    ``start_period`` moves this ceiling with it. The host term is read, never
    asserted: ``probe_host_conditions`` takes its readers as keyword seams that
    argv cannot reach, for the same reason the prod-promotion gate had its
    caller-assertable health surface deleted under OMN-18319.
    """
    multiplier = deps_contention_multiplier(host.saturation)
    scaled = int(model_budget.timeout_seconds * multiplier)
    clamped = scaled > DEPS_COMPOSE_UP_HARD_UPPER_BOUND_SECONDS
    return ModelDepsRecreateBudget(
        timeout_seconds=min(scaled, DEPS_COMPOSE_UP_HARD_UPPER_BOUND_SECONDS),
        hard_upper_bound_seconds=DEPS_COMPOSE_UP_HARD_UPPER_BOUND_SECONDS,
        model_seconds=model_budget.timeout_seconds,
        host_multiplier=multiplier,
        saturation=host.saturation,
        source_service=model_budget.source_service,
        clamped_to_hard_upper_bound=clamped,
    )


def lane_is_mid_recreate(
    expected_services: Sequence[str],
    states: Mapping[str, tuple[str, int | None]],
) -> bool:
    """True when any expected service is absent or not running.

    ABSENT COUNTS, and counts first. The 2026-09-18 lane had no redpanda
    container at all -- not a stopped one, not a created one -- so a check that
    only inspected the states of containers it could see would have read that
    lane as quiescent and authorised the kill that produced it.
    """
    for service in expected_services:
        state, _exit_code = states.get(service, ("missing", None))
        if state != "running":
            return True
    return False


def defer_until_host_quiesces(
    *,
    host_probe: HostProbe | None = None,
    threshold: float = DEPS_RECREATE_DEFER_SATURATION,
    max_wait_seconds: int = DEPS_RECREATE_DEFER_MAX_WAIT_SECONDS,
    poll_seconds: int = DEPS_RECREATE_DEFER_POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[float, ModelHostConditions]:
    """Wait for the machine before the first removal, or refuse having done none.

    Returns ``(seconds_waited, final_host_reading)``. Raises
    ``HostContentionDeferralError`` when the host is still above ``threshold``
    after ``max_wait_seconds``.

    An UNREADABLE machine does not defer. That is the one place this module
    reads a degraded value inward rather than outward, and it is deliberate:
    deferring on an unreadable ``loadavg`` would refuse every recreate on any
    host whose ``/proc`` this process cannot read, which converts a monitoring
    gap into a total deploy outage. The ceiling's UNKNOWN term already charges
    that same reading the worst case, so an unreadable host still gets the
    widest ceiling -- it just does not get refused at the door.
    """
    probe = host_probe or probe_host_conditions
    started = monotonic()
    host = probe()
    if host.saturation is None or host.saturation <= threshold:
        return 0.0, host

    deadline = started + max_wait_seconds
    while monotonic() < deadline:
        logger.warning(
            "deps recreate DEFERRED: %s exceeds the %.2f start threshold; "
            "waiting (nothing has been stopped, created or removed)",
            host.describe(),
            threshold,
        )
        sleep(poll_seconds)
        host = probe()
        if host.saturation is None or host.saturation <= threshold:
            waited = monotonic() - started
            logger.info(
                "deps recreate proceeding after a %.0fs deferral: %s",
                waited,
                host.describe(),
            )
            return waited, host

    waited = monotonic() - started
    raise HostContentionDeferralError(
        f"{EnumRecreateOutcome.DEFERRED_HOST_CONTENTION.value}: refused to start "
        f"a deps recreate after waiting {waited:.0f}s for the host to fall to "
        f"saturation {threshold:.2f}; last reading {host.describe()}. THE LANE "
        f"WAS NOT TOUCHED -- no container was stopped, created or removed by "
        f"this command, so the deps are exactly as this job found them."
    )


def supervise_compose_up(
    cmd: list[str],
    *,
    phase: str,
    expected_services: Sequence[str],
    budget: ModelDepsRecreateBudget,
    state_reader: StateReader,
    spawn: Callable[[list[str]], subprocess.Popen[str]],
    progress_reader: ProgressReader | None = None,
    deferred_seconds: float = 0.0,
    poll_seconds: float = SUPERVISOR_POLL_INTERVAL_SECONDS,
    stall_seconds: float = MID_RECREATE_STALL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ModelRecreateSupervision:
    """Run a deps compose-up under a ceiling that cannot cancel a live recreate.

    LIVENESS IS READ FROM TWO PLACES, NOT ONE, and the second is not optional
    in practice. A container being pulled, created or started sits in ONE
    compose state for the whole of that work: a redpanda image pull under load
    can show ``created`` and nothing else for ten minutes. A stall detector
    watching only container states would therefore read a perfectly healthy
    recreate as wedged and cancel it -- which is this ticket's defect with a
    longer fuse. So progress is "the lane changed OR the compose CLI wrote
    something", and ``DeployExecutor`` supplies the second by handing back the
    size of the file the child is writing to.

    THE CALLER OWNS THE CHILD'S STREAMS. ``spawn`` must NOT hand back a process
    whose output goes to an unread ``PIPE``: this supervisor deliberately never
    reads from the child while it runs, so a pipe would fill and deadlock the
    very recreate it exists to let finish. ``DeployExecutor`` spawns against a
    temporary file and reads it afterwards.

    The returned record is EVIDENCE, not a verdict: the caller decides the
    phase outcome from live container state afterwards, exactly as
    ``_compose_up`` already does for the runtime phase. What this guarantees is
    the property the ticket names -- the command is not cancelled while the
    lane is mid-recreate and still changing.
    """
    baseline = _read_states(state_reader)
    started = monotonic()
    process = spawn(cmd)

    read_progress = progress_reader or (lambda: 0)
    anchor: float | None = None
    last_states = baseline
    last_progress = _read_progress(read_progress)
    last_change = started
    warned_past_ceiling = False

    def _finish(
        outcome: EnumRecreateOutcome,
        *,
        returncode: int | None,
        states: Mapping[str, tuple[str, int | None]],
    ) -> ModelRecreateSupervision:
        now = monotonic()
        elapsed_total = now - started
        elapsed_anchored = now - (anchor if anchor is not None else started)
        return ModelRecreateSupervision(
            phase=phase,
            outcome=outcome,
            ceiling_seconds=budget.timeout_seconds,
            hard_upper_bound_seconds=budget.hard_upper_bound_seconds,
            elapsed_seconds=round(elapsed_total, 1),
            anchored_elapsed_seconds=round(elapsed_anchored, 1),
            waited_past_ceiling_seconds=round(
                max(0.0, elapsed_anchored - budget.timeout_seconds), 1
            ),
            deferred_seconds=round(deferred_seconds, 1),
            anchored_at_first_container_change=anchor is not None,
            mid_recreate_at_decision=lane_is_mid_recreate(expected_services, states),
            returncode=returncode,
            budget_description=budget.describe(),
        )

    while True:
        # POLLED FIRST, EVERY TIME ROUND, AND BEFORE ANY DECISION. The loop
        # sleeps at its END rather than its start so that a command which
        # finished during the sleep is seen as finished, not as a lane to
        # adjudicate. With the sleep first, a recreate that completed at the
        # same instant the ceiling elapsed was signalled anyway -- a cancel
        # delivered to a process that had already succeeded, which is this
        # ticket's defect reproduced by its own fix.
        returncode = process.poll()
        if returncode is not None:
            return _finish(
                EnumRecreateOutcome.COMPLETED,
                returncode=returncode,
                states=_read_states(state_reader),
            )

        now = monotonic()
        states = _read_states(state_reader)

        progress = _read_progress(read_progress)
        if progress > last_progress:
            last_progress = progress
            last_change = now

        if states != last_states:
            last_states = states
            last_change = now
            if anchor is None:
                # The FIRST observed change to the lane is the anchor: before
                # it, compose has not touched a container and the elapsed time
                # is queueing, not recreating. Spending the recreate's budget
                # on the queue is what fired a 300s ceiling during a removal.
                anchor = now
                logger.info(
                    "phase %s: first container-state change observed %.0fs after "
                    "the command started; the %ss ceiling is measured from here",
                    phase,
                    now - started,
                    budget.timeout_seconds,
                )

        elapsed = now - (anchor if anchor is not None else started)
        if elapsed <= budget.timeout_seconds:
            sleep(poll_seconds)
            continue

        mid_recreate = lane_is_mid_recreate(expected_services, states)
        if not mid_recreate:
            # Every expected service is running and compose has still not
            # returned. Nothing is mid-removal, so ending the command cannot
            # destroy the lane; the caller's verify + recovery decides the
            # verdict from live state, as it does for a non-zero exit.
            logger.warning(
                "phase %s exceeded its %ss ceiling with every expected service "
                "running; ending the command (nothing is mid-removal)",
                phase,
                budget.timeout_seconds,
            )
            _terminate(process)
            return _finish(
                EnumRecreateOutcome.ENDED_LANE_SETTLED,
                returncode=process.returncode,
                states=states,
            )

        if now - last_change >= stall_seconds:
            logger.error(
                "phase %s is mid-recreate and has not changed for %.0fs; "
                "treating the recreate as wedged rather than slow",
                phase,
                now - last_change,
            )
            _terminate(process)
            return _finish(
                EnumRecreateOutcome.KILLED_MID_RECREATE_STALLED,
                returncode=process.returncode,
                states=states,
            )

        if elapsed >= budget.hard_upper_bound_seconds:
            logger.error(
                "phase %s reached the %ss hard upper bound while still "
                "mid-recreate; ending the command -- the caller MUST converge "
                "the deps before anything else runs",
                phase,
                budget.hard_upper_bound_seconds,
            )
            _terminate(process)
            return _finish(
                EnumRecreateOutcome.KILLED_HARD_UPPER_BOUND,
                returncode=process.returncode,
                states=states,
            )

        if not warned_past_ceiling:
            warned_past_ceiling = True
            logger.warning(
                "phase %s is past its %ss ceiling and the lane is MID-RECREATE; "
                "WAITING rather than cancelling -- cancelling here is what "
                "removed the dev-lane broker on 2026-09-18 (OMN-18692). The "
                "wait ends when the lane stops changing for %.0fs or at the "
                "%ss hard upper bound.",
                phase,
                budget.timeout_seconds,
                stall_seconds,
                budget.hard_upper_bound_seconds,
            )

        sleep(poll_seconds)


def _read_progress(read_progress: ProgressReader) -> int:
    """Read the child's output size, or 0 when it cannot be read.

    Never raises: an unreadable progress signal degrades this to the
    container-state signal alone, which is a narrower liveness test but not a
    broken one. Returning 0 is safe because the comparison is strictly
    greater-than against the last reading, so a reader that starts failing
    simply stops contributing rather than manufacturing progress.
    """
    try:
        return int(read_progress())
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        logger.warning("recreate supervisor: could not read compose progress: %s", exc)
        return 0


def _read_states(
    state_reader: StateReader,
) -> Mapping[str, tuple[str, int | None]]:
    """Read the lane, or report an unreadable lane as an EMPTY reading.

    An unreadable lane must never read as a settled one: ``lane_is_mid_recreate``
    treats a service it cannot see as mid-recreate, so an empty mapping is the
    conservative value here and a docker daemon this process cannot reach can
    never authorise a kill.
    """
    try:
        return dict(state_reader())
    except Exception as exc:  # noqa: BLE001 -- an unreadable lane is not a crash
        logger.warning("recreate supervisor: could not read lane state: %s", exc)
        return {}


def _terminate(process: subprocess.Popen[str]) -> None:
    """End the compose CLI, SIGTERM first so it can unwind its own work.

    SIGTERM rather than SIGKILL because the compose CLI handles it: it stops
    issuing new daemon calls and returns, which is a materially better ending
    for a half-issued recreate than having the process vanish. The SIGKILL
    below is the backstop for a CLI that ignores the first signal.

    This function is only ever reached on the two named endings that are NOT
    "the lane is mid-recreate and moving" -- see the caller.
    """
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
