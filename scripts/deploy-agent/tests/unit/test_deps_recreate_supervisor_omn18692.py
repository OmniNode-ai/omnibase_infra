# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deps recreate is never cancelled mid-removal (OMN-18692).

THE FIXTURE IS THE INCIDENT. Every number below is read from the 2026-09-18
timeline recorded in ``docs/tracking/ROLLING_WORK_LEDGER.md:4256`` and from the
deploy agent's own journal on ``192.168.86.201``:

* 12:55:04Z -- job ``6cee06d3-7337-4263-a12b-851a46e0bf01`` accepted, scope=full,
  host ``load1 28.42`` on 32 cores (saturation 0.888)
* 13:01:56Z -- dockerd ``Canceled: context canceled`` on four
  ``redpandadata/redpanda:v24.2.7`` containers
* 13:08:33Z -- postgres half-recreated as
  ``7a423d93d83e_omnibase-infra-postgres``, state ``created``
* 13:13:29Z -- "docker compose up exceeded its 300s ceiling for phase core and
  was killed with the lane mid-recreate"
* 13:17:31Z -- ``deploy-agent-dev.service`` failed, restart counter 7,
  crash-looping on ``NoBrokersAvailable`` against the broker it had just
  destroyed
* 13:35:56Z -- an operator's hand ``up -d postgres redpanda valkey`` returns 0;
  all three Up healthy 32s later

The recreate that was killed was HEALTHY, not hung: once it was allowed to
finish -- which took an operator, because the agent had destroyed its own
control bus -- the same work completed. So the fixture models a recreate that
takes about 22 minutes and succeeds, and the property under test is that
nothing cancels it.

RED CONTROL. ``test_the_flat_bound_fires_while_the_lane_is_mid_removal``
computes the pre-change mechanism against this fixture from the live
``PHASE_TIMEOUTS[Phase.CORE]`` constant and shows it lands inside the removal
window. It is not a historical note: it fails if someone "fixes" this by
widening that constant, because a wider flat bound still cannot see the
machine and would fire mid-removal on a slower host.

POSITIVE CONTROL. A test asserting "no cancel" is worthless if the harness
cannot produce a cancel, so ``test_a_stalled_recreate_is_ended`` drives the
same machinery to a termination. Both tests read the same fake process.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from deploy_agent.compose_budget import ModelPhaseBudget
from deploy_agent.events import EnumRecreateOutcome, Phase
from deploy_agent.executor import PHASE_TIMEOUTS
from deploy_agent.host_conditions import EnumBuildCacheState, ModelHostConditions
from deploy_agent.recreate_supervisor import (
    DEPS_COMPOSE_UP_FLOOR_SECONDS,
    DEPS_COMPOSE_UP_MARGIN_SECONDS,
    MID_RECREATE_STALL_SECONDS,
    HostContentionDeferralError,
    defer_until_host_quiesces,
    deps_contention_multiplier,
    derive_deps_recreate_budget,
    lane_is_mid_recreate,
    supervise_compose_up,
)

pytestmark = pytest.mark.unit

DEPS = ("postgres", "redpanda", "valkey")

# The machine, as measured: load1 28.42 over 32 cores.
INCIDENT_LOAD1 = 28.42
INCIDENT_CPUS = 32
INCIDENT_SATURATION = INCIDENT_LOAD1 / INCIDENT_CPUS  # 0.888

# The fixture's timeline, in seconds from the moment compose was invoked.
REMOVAL_STARTS_AT = 120.0
POSTGRES_BACK_AT = 600.0
VALKEY_BACK_AT = 900.0
REDPANDA_BACK_AT = 1320.0  # ~22 minutes, the measured accept-to-recovery span
COMMAND_EXITS_AT = REDPANDA_BACK_AT


class _Clock:
    """A clock the supervisor's own seams drive; ``sleep`` is the only advance."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeComposeProcess:
    """A compose CLI that succeeds at ``exits_at`` unless it is signalled first.

    ``terminate`` is recorded rather than ignored, because "was the recreate
    cancelled" is the entire question this module asks. A terminated process
    returns a signal code, exactly as the real one does, so a caller cannot
    read a cancellation as a clean exit.
    """

    def __init__(self, clock: _Clock, *, exits_at: float) -> None:
        self._clock = clock
        self._exits_at = exits_at
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.terminated_at: float | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self._clock.now >= self._exits_at:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.terminated_at = self._clock.now
        self.returncode = -15

    def kill(self) -> None:  # pragma: no cover - the fake always honours SIGTERM
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


def _incident_lane(clock: _Clock) -> Mapping[str, tuple[str, int | None]]:
    """The lane's container states at ``clock.now``, as the incident had them.

    Before the removal every dep is running. The removal takes all three away
    at once -- which is what compose does, and what left ``docker ps -a``
    with no ``omnibase-infra-redpanda`` row at all -- and they come back one at
    a time.
    """
    now = clock.now
    if now < REMOVAL_STARTS_AT:
        return dict.fromkeys(DEPS, ("running", None))
    states: dict[str, tuple[str, int | None]] = {}
    if now >= POSTGRES_BACK_AT:
        states["postgres"] = ("running", None)
    elif now >= REMOVAL_STARTS_AT:
        # The half-recreated `7a423d93d83e_omnibase-infra-postgres`.
        states["postgres"] = ("created", None)
    if now >= VALKEY_BACK_AT:
        states["valkey"] = ("running", None)
    if now >= REDPANDA_BACK_AT:
        states["redpanda"] = ("running", None)
    return states


def _incident_budget() -> object:
    """The ceiling this change derives for the incident's compose model + host."""
    model = ModelPhaseBudget(
        # postgres declares `start_period: 180s` and is gated on via
        # `depends_on: {condition: service_healthy}` in
        # docker/docker-compose.infra.yml -- read live by
        # test_executor_deps_recreate_omn18692.py, restated here so this module
        # exercises the supervisor without opening the compose files.
        timeout_seconds=max(
            DEPS_COMPOSE_UP_FLOOR_SECONDS, 180 + DEPS_COMPOSE_UP_MARGIN_SECONDS
        ),
        floor_seconds=DEPS_COMPOSE_UP_FLOOR_SECONDS,
        margin_seconds=DEPS_COMPOSE_UP_MARGIN_SECONDS,
        source_service="postgres",
        source_start_period_seconds=180,
        gating_services=DEPS,
        compose_files=("docker-compose.infra.yml",),
    )
    return derive_deps_recreate_budget(model, _incident_host())


def _incident_host() -> ModelHostConditions:
    return ModelHostConditions(
        load1=INCIDENT_LOAD1,
        cpu_count=INCIDENT_CPUS,
        saturation=INCIDENT_SATURATION,
        cache_state=EnumBuildCacheState.WARM,
        contention_multiplier=1.0,
        cache_multiplier=1.0,
    )


def test_the_flat_bound_fires_while_the_lane_is_mid_removal() -> None:
    """RED CONTROL: the pre-change mechanism kills this fixture mid-removal.

    Reads the live constant rather than the literal 300, so widening the
    constant instead of fixing the mechanism does not quietly satisfy this --
    the assertion is about WHERE the bound lands relative to the removal, not
    about its value.
    """
    flat_bound = PHASE_TIMEOUTS[Phase.CORE]
    clock = _Clock()
    clock.now = float(flat_bound)
    states = _incident_lane(clock)

    assert REMOVAL_STARTS_AT < flat_bound < REDPANDA_BACK_AT
    assert lane_is_mid_recreate(DEPS, states)
    assert "redpanda" not in states, (
        "the fixture must reproduce the measured residue: at the moment the "
        "flat bound fires there is no redpanda container at all"
    )


def test_the_22_minute_deps_recreate_is_never_cancelled_omn18692() -> None:
    """AC1: the supervisor waits out a live recreate rather than cancelling it.

    The ceiling IS exceeded here -- that is the point. What the supervisor must
    not do is turn exceeding it into a signal while containers are missing.
    """
    clock = _Clock()
    process = _FakeComposeProcess(clock, exits_at=COMMAND_EXITS_AT)
    budget = _incident_budget()

    supervision = supervise_compose_up(
        ["docker", "compose", "up", "-d", "--force-recreate"],
        phase=Phase.CORE.value,
        expected_services=DEPS,
        budget=budget,
        state_reader=lambda: _incident_lane(clock),
        spawn=lambda _argv: process,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert process.terminate_calls == 0, (
        "the recreate was cancelled; this is the 2026-09-18 defect -- dockerd "
        "logged 'Canceled: context canceled' on four redpanda containers and "
        "the lane lost its broker"
    )
    assert process.kill_calls == 0
    assert supervision.outcome is EnumRecreateOutcome.COMPLETED
    assert supervision.returncode == 0

    # The falsifier the ticket names: no container removed without its
    # replacement completing. Read from the lane, not from the supervisor.
    final = _incident_lane(clock)
    assert not lane_is_mid_recreate(DEPS, final)
    assert all(final[service] == ("running", None) for service in DEPS)

    # The incident's own 22 minutes now fit INSIDE the derived ceiling, which
    # is the fix: the flat bound it was killed by is a fifth of what the
    # compose model and the machine together say this work is worth. Stated as
    # an assertion rather than left implicit, because it is the claim.
    assert supervision.anchored_elapsed_seconds < supervision.ceiling_seconds
    assert supervision.ceiling_seconds > PHASE_TIMEOUTS[Phase.CORE] * 4


def test_a_recreate_that_outruns_even_the_derived_ceiling_is_not_cancelled() -> None:
    """AC1: past the derived ceiling, a LIVE recreate is still waited out.

    The test above proves the ceiling is now wide enough for the measured
    incident. This one proves the ceiling is not the mechanism -- a slower
    host, a bigger volume or a colder image would put the same healthy work
    past any number, and the property that matters is what happens THEN.
    """
    clock = _Clock()
    budget = _incident_budget()
    # Same shape as the incident, stretched past the derived ceiling: the lane
    # keeps making progress the whole way, one service at a time.
    exits_at = budget.timeout_seconds + REMOVAL_STARTS_AT + 600.0
    process = _FakeComposeProcess(clock, exits_at=exits_at)
    marks = (
        REMOVAL_STARTS_AT,
        exits_at * 0.4,
        exits_at * 0.7,
        exits_at,
    )

    # The compose CLI is writing progress lines the whole time, which is the
    # liveness signal a long image pull gives when no container state moves.
    def _compose_output_bytes() -> int:
        return int(clock.now * 40)

    def _slow_lane() -> Mapping[str, tuple[str, int | None]]:
        now = clock.now
        if now < marks[0]:
            return dict.fromkeys(DEPS, ("running", None))
        states: dict[str, tuple[str, int | None]] = {"postgres": ("created", None)}
        if now >= marks[1]:
            states["postgres"] = ("running", None)
        if now >= marks[2]:
            states["valkey"] = ("running", None)
        if now >= marks[3]:
            states["redpanda"] = ("running", None)
        return states

    supervision = supervise_compose_up(
        ["docker", "compose", "up", "-d", "--force-recreate"],
        phase=Phase.CORE.value,
        expected_services=DEPS,
        budget=budget,
        state_reader=_slow_lane,
        spawn=lambda _argv: process,
        progress_reader=_compose_output_bytes,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert process.terminate_calls == 0
    assert supervision.outcome is EnumRecreateOutcome.COMPLETED
    assert supervision.waited_past_ceiling_seconds > 0, (
        "the fixture must actually cross the ceiling, or it proves nothing "
        "about what the supervisor does past one"
    )
    assert not lane_is_mid_recreate(DEPS, _slow_lane())


def test_the_ceiling_is_anchored_at_the_first_container_change() -> None:
    """AC2: the clock starts when compose touches the lane, not when it starts.

    The 120s this fixture spends before the first removal is queueing on a
    loaded host. Charging it to the recreate is how a bound gets spent before
    the work it bounds begins.
    """
    clock = _Clock()
    process = _FakeComposeProcess(clock, exits_at=COMMAND_EXITS_AT)
    budget = _incident_budget()

    supervision = supervise_compose_up(
        ["docker", "compose", "up"],
        phase=Phase.CORE.value,
        expected_services=DEPS,
        budget=budget,
        state_reader=lambda: _incident_lane(clock),
        spawn=lambda _argv: process,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert supervision.anchored_at_first_container_change is True
    assert supervision.elapsed_seconds >= COMMAND_EXITS_AT
    # The difference between the two clocks is the queueing, and it is the
    # quantity the flat bound was unknowingly charging against the recreate.
    queued = supervision.elapsed_seconds - supervision.anchored_elapsed_seconds
    assert queued == pytest.approx(REMOVAL_STARTS_AT, abs=10.0)


def test_the_ceiling_scales_with_the_measured_host_load() -> None:
    """AC2: the incident's saturation widens the ceiling; an idle host does not."""
    idle = ModelHostConditions(
        load1=1.0,
        cpu_count=32,
        saturation=1.0 / 32,
        cache_state=EnumBuildCacheState.WARM,
        contention_multiplier=1.0,
        cache_multiplier=1.0,
    )
    model = ModelPhaseBudget(
        timeout_seconds=DEPS_COMPOSE_UP_FLOOR_SECONDS,
        floor_seconds=DEPS_COMPOSE_UP_FLOOR_SECONDS,
        margin_seconds=DEPS_COMPOSE_UP_MARGIN_SECONDS,
        source_service="postgres",
        source_start_period_seconds=180,
        gating_services=DEPS,
        compose_files=(),
    )

    quiet = derive_deps_recreate_budget(model, idle)
    loaded = derive_deps_recreate_budget(model, _incident_host())

    assert quiet.host_multiplier == 1.0
    assert loaded.host_multiplier > 1.0, (
        "saturation 0.888 must widen the deps ceiling; the build term's 1.0 "
        "threshold does not engage below 1.0 and would have left this "
        "incident's ceiling exactly where it was"
    )
    assert loaded.timeout_seconds > quiet.timeout_seconds
    assert "saturation 0.89" in loaded.describe()


def test_an_unreadable_machine_widens_rather_than_narrows() -> None:
    """A reading that failed is charged the worst case it might be hiding."""
    assert deps_contention_multiplier(None) > deps_contention_multiplier(0.0)


def test_a_missing_container_reads_as_mid_recreate() -> None:
    """ABSENT is the state the incident actually left, and it must count.

    ``docker ps -a`` had no ``omnibase-infra-redpanda`` row at all, so a check
    that only inspected the containers it could see would have read that lane
    as settled and authorised the cancel that produced it.
    """
    assert lane_is_mid_recreate(DEPS, {"postgres": ("running", None)}) is True
    assert lane_is_mid_recreate(DEPS, dict.fromkeys(DEPS, ("running", None))) is False
    assert (
        lane_is_mid_recreate(
            DEPS,
            {**dict.fromkeys(DEPS, ("running", None)), "redpanda": ("created", None)},
        )
        is True
    )


def test_an_unreadable_lane_never_authorises_a_cancel() -> None:
    """A docker daemon this process cannot reach reads as mid-recreate.

    The supervisor's state reader swallows and returns an empty mapping, and an
    empty mapping is mid-recreate by the rule above -- so a lane the agent
    cannot see can never be the reason it cancels a live recreate.
    """
    clock = _Clock()
    process = _FakeComposeProcess(clock, exits_at=COMMAND_EXITS_AT)

    def _unreadable() -> Mapping[str, tuple[str, int | None]]:
        raise RuntimeError("Cannot connect to the Docker daemon")

    supervision = supervise_compose_up(
        ["docker", "compose", "up"],
        phase=Phase.CORE.value,
        expected_services=DEPS,
        budget=_incident_budget(),
        state_reader=_unreadable,
        spawn=lambda _argv: process,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert process.terminate_calls == 0
    assert supervision.outcome is EnumRecreateOutcome.COMPLETED


def test_a_stalled_recreate_is_ended() -> None:
    """POSITIVE CONTROL: the harness CAN produce a cancel, on a wedged lane.

    Without this, "no cancel" above could be satisfied by a supervisor that
    never signals anything, which would be a different defect with the same
    green test. A lane whose container state has not moved for the declared
    stall window is read as wedged rather than slow, and that ending carries
    its own token so it is never confused with the one above.
    """
    clock = _Clock()
    # A process that never exits, against a lane frozen mid-removal.
    process = _FakeComposeProcess(clock, exits_at=float("inf"))
    frozen = {"postgres": ("created", None)}

    supervision = supervise_compose_up(
        ["docker", "compose", "up"],
        phase=Phase.CORE.value,
        expected_services=DEPS,
        budget=_incident_budget(),
        state_reader=lambda: frozen,
        spawn=lambda _argv: process,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert process.terminate_calls == 1
    assert supervision.outcome is EnumRecreateOutcome.KILLED_MID_RECREATE_STALLED
    assert supervision.mid_recreate_at_decision is True
    assert clock.now >= MID_RECREATE_STALL_SECONDS


def test_a_settled_lane_past_its_ceiling_is_ended_without_the_stall_wait() -> None:
    """Nothing is mid-removal, so ending the command cannot destroy the lane."""
    clock = _Clock()
    process = _FakeComposeProcess(clock, exits_at=float("inf"))
    running = dict.fromkeys(DEPS, ("running", None))
    budget = _incident_budget()

    supervision = supervise_compose_up(
        ["docker", "compose", "up"],
        phase=Phase.CORE.value,
        expected_services=DEPS,
        budget=budget,
        state_reader=lambda: running,
        spawn=lambda _argv: process,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert supervision.outcome is EnumRecreateOutcome.ENDED_LANE_SETTLED
    assert supervision.mid_recreate_at_decision is False
    # Ended at the ceiling, NOT after the stall window -- a settled lane does
    # not buy the extra wait a live recreate does.
    assert clock.now < budget.timeout_seconds + MID_RECREATE_STALL_SECONDS


def test_the_hard_upper_bound_ends_a_recreate_that_never_finishes() -> None:
    """ "Adapts to the host" must not mean "unbounded" (the build budget's rule).

    A lane that keeps changing forever would otherwise hold the supervisor for
    ever. This ending is named separately from the stall so a reader can tell a
    wedged daemon from one that is merely slower than any budget.
    """
    clock = _Clock()
    process = _FakeComposeProcess(clock, exits_at=float("inf"))
    budget = _incident_budget()
    tick = {"n": 0}

    def _always_changing() -> Mapping[str, tuple[str, int | None]]:
        tick["n"] += 1
        # Never all-running, and never the same twice: progress that never
        # arrives.
        return {"postgres": (f"created-{tick['n']}", None)}

    supervision = supervise_compose_up(
        ["docker", "compose", "up"],
        phase=Phase.CORE.value,
        expected_services=DEPS,
        budget=budget,
        state_reader=_always_changing,
        spawn=lambda _argv: process,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert supervision.outcome is EnumRecreateOutcome.KILLED_HARD_UPPER_BOUND
    assert clock.now >= budget.hard_upper_bound_seconds


def test_a_loaded_host_defers_the_deps_recreate_before_any_removal_omn18692() -> None:
    """AC1's second branch: refuse to START, having mutated nothing.

    The refusal is a named class, not a generic ``RuntimeError``, because the
    fact that distinguishes it from every other deploy failure is that the lane
    was not touched.
    """
    clock = _Clock()
    wedged = ModelHostConditions(
        load1=96.0,
        cpu_count=32,
        saturation=3.0,
        cache_state=EnumBuildCacheState.WARM,
        contention_multiplier=1.0,
        cache_multiplier=1.0,
    )

    with pytest.raises(HostContentionDeferralError) as excinfo:
        defer_until_host_quiesces(
            host_probe=lambda: wedged,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    message = str(excinfo.value)
    assert EnumRecreateOutcome.DEFERRED_HOST_CONTENTION.value in message
    assert "THE LANE WAS NOT TOUCHED" in message
    assert "load1 96.00" in message


def test_a_host_that_quiesces_proceeds_and_the_wait_is_reported() -> None:
    """A deferral that ends in a start still records how long it held."""
    clock = _Clock()
    readings = iter(
        [
            ModelHostConditions(
                load1=96.0,
                cpu_count=32,
                saturation=3.0,
                cache_state=EnumBuildCacheState.WARM,
                contention_multiplier=1.0,
                cache_multiplier=1.0,
            ),
            ModelHostConditions(
                load1=8.0,
                cpu_count=32,
                saturation=0.25,
                cache_state=EnumBuildCacheState.WARM,
                contention_multiplier=1.0,
                cache_multiplier=1.0,
            ),
        ]
    )

    waited, host = defer_until_host_quiesces(
        host_probe=lambda: next(readings),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert waited > 0
    assert host.saturation == 0.25


def test_an_unreadable_load_average_does_not_defer() -> None:
    """The one inward reading, and it is deliberate.

    Deferring on an unreadable ``/proc/loadavg`` would refuse every recreate on
    any host this process cannot read, turning a monitoring gap into a total
    deploy outage. The ceiling's UNKNOWN term still charges that same reading
    the widest multiplier, so such a host gets the widest ceiling -- it just is
    not refused at the door.
    """
    clock = _Clock()
    unreadable = ModelHostConditions(
        load1=None,
        cpu_count=None,
        saturation=None,
        cache_state=EnumBuildCacheState.UNKNOWN,
        contention_multiplier=1.0,
        cache_multiplier=1.0,
    )

    waited, _host = defer_until_host_quiesces(
        host_probe=lambda: unreadable,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert waited == 0.0
