# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``consumer_sync`` readiness dimension, and the chain it has to complete.

Measuring a wedge is not the deliverable. The deliverable is that ``docker ps``
says ``unhealthy``, because that is the single fact the deploy agent's AC7
force-recreate probes and the only one an operator sees without reading JSON.
So these tests walk the whole chain and not just its first link:

    supervisor measures  ->  bus exposes  ->  monitor renders a dimension
        ->  /health body carries it  ->  container healthcheck exits non-zero

The last two links already existed; ``container_healthcheck`` has read the
dimension block since OMN-15217. What follows asserts they are actually
reached, because a dimension that is computed and then folded into nothing is
exactly the shape of the three green surfaces this ticket is about.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)
from omnibase_infra.models.health.model_consumer_sync_status import (
    ModelConsumerSyncStatus,
)
from omnibase_infra.runtime.health.container_healthcheck import (
    evaluate_health_response,
)
from omnibase_infra.runtime.health.runtime_health_block import (
    build_runtime_health_block,
    fold_runtime_verdict_into_status,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
    describe_consumer_sync,
)

pytestmark = pytest.mark.unit

TOPIC = "onex.cmd.omnimarket.occ-autobind.v1"  # onex-topic-allow: replay of a recorded incident
GROUP = "local.omnimarket.pr_lifecycle_fix_effect.consume.1.0.0.__i.runtime-effects"


def _status(
    *,
    ready: bool,
    backlog: int = 0,
    stalled_seconds: float = 0.0,
    silent_for: float = 0.0,
    reason: EnumConsumerStallReason = EnumConsumerStallReason.NOT_STALLED_IDLE,
    last_rejoin_failed: bool = False,
    evaluated: bool = True,
) -> ModelConsumerSyncStatus:
    """Build a status directly.

    The supervisor's own tests prove these values are measured; this file is
    about what the monitor does with them, so constructing them here keeps the
    two concerns from hiding each other's failures.
    """
    status = ModelConsumerSyncStatus(
        topic=TOPIC,
        consumer_group=GROUP,
        evaluated=evaluated,
        reason=reason,
        backlog_records=backlog,
        seconds_since_last_record=silent_for,
        stalled_seconds=stalled_seconds,
        assigned_partitions=1,
        broker_reachable=True,
        rejoin_count=1 if last_rejoin_failed else 0,
        last_rejoin_failed=last_rejoin_failed,
        unready_after_seconds=600.0,
    )
    assert status.ready is ready, (
        "the fixture's own expectation disagrees with the derived property"
    )
    return status


class _FakeBus:
    """An event bus that can answer the consumer-sync question."""

    def __init__(self, *statuses: ModelConsumerSyncStatus) -> None:
        self._statuses = statuses

    def consumer_sync_statuses(self) -> tuple[ModelConsumerSyncStatus, ...]:
        return self._statuses


class _SilentBus:
    """An event bus with no consumer-sync surface at all (the in-memory bus)."""


def _manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.total_discovered = 0
    manifest.total_errors = 0
    manifest.errors = ()
    manifest.all_subscribe_topics.return_value = ()
    return manifest


async def _run_once(bus: object) -> object:
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=bus,  # type: ignore[arg-type]
        bootstrap_servers="",
        check_interval_seconds=300.0,
        boot_grace_seconds=0.0,
    )
    with (
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=_manifest(),
        ),
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._filter_manifest_for_runtime_profile",
            side_effect=lambda manifest: manifest,
        ),
    ):
        return await monitor.run_once()


def _dimension(event: object, name: str) -> object | None:
    for dimension in event.dimensions:  # type: ignore[attr-defined]
        if dimension.name == name:
            return dimension
    return None


# --- the dimension ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stalled_group_makes_the_dimension_critical() -> None:
    """CRITICAL, not DEGRADED, and the reason is not severity theatre.

    DEGRADED would be read as "partly working". A runtime holding a group it
    is not draining is not partly doing that topic's work; it is doing none of
    it while committing nothing, and every record published meanwhile is
    waiting on a client that will not come back on its own. CRITICAL is also
    the only grade that is fail-closed against the healthcheck's
    ``--degraded-policy`` being set the other way on some lane.
    """
    event = await _run_once(
        _FakeBus(
            _status(
                ready=False,
                backlog=16,
                stalled_seconds=3_000.0,
                silent_for=3_000.0,
                reason=EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING,
            )
        )
    )
    dimension = _dimension(event, "consumer_sync")
    assert dimension is not None, (
        "the runtime publishes no consumer_sync dimension at all -- the wedge "
        "of 2026-09-19T04:51Z would be invisible here exactly as it was then"
    )
    assert dimension.status == "CRITICAL"  # type: ignore[attr-defined]
    assert event.status == "CRITICAL"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_the_detail_carries_the_lag_and_the_last_advance() -> None:
    """Evidence, not a verdict. An operator must not have to go and measure.

    The two numbers are the two that disagreed with every green surface during
    both outages: the backlog against the leaders, and how long it has been
    since a record moved.
    """
    event = await _run_once(
        _FakeBus(
            _status(
                ready=False,
                backlog=16,
                stalled_seconds=3_000.0,
                silent_for=3_000.0,
                reason=EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING,
            )
        )
    )
    detail = _dimension(event, "consumer_sync").detail  # type: ignore[union-attr]
    assert "16" in detail, "the measured lag must appear in the evidence"
    assert "3000" in detail.replace(",", "") or "3000s" in detail.replace(",", "")
    assert GROUP.split(".__i.")[0] in detail, "the group must be named"


@pytest.mark.asyncio
async def test_healthy_groups_report_the_lag_they_measured() -> None:
    """A green dimension still carries its numbers.

    A dimension whose detail is only ever interesting when it is red is one
    nobody can use to tell "measured and fine" from "not measured".
    """
    event = await _run_once(_FakeBus(_status(ready=True, backlog=0, silent_for=42.0)))
    dimension = _dimension(event, "consumer_sync")
    assert dimension.status == "HEALTHY"  # type: ignore[union-attr]
    assert "1 consumer group" in dimension.detail  # type: ignore[union-attr]
    assert event.status == "HEALTHY"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_bus_with_no_consumer_sync_surface_is_healthy_not_absent() -> None:
    """The in-memory bus, and every non-Kafka transport.

    Reported HEALTHY with a detail that says the surface was not available,
    rather than omitted: a dimension that disappears is indistinguishable from
    a dimension that was never added, and the next reader has to go and read
    the source to tell which.
    """
    event = await _run_once(_SilentBus())
    dimension = _dimension(event, "consumer_sync")
    assert dimension is not None
    assert dimension.status == "HEALTHY"  # type: ignore[union-attr]
    assert "not available" in dimension.detail.lower()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_one_wedged_group_among_healthy_ones_is_enough() -> None:
    """Any, not all. The runtime owns every group it holds."""
    event = await _run_once(
        _FakeBus(
            _status(ready=True, backlog=0),
            _status(
                ready=False,
                backlog=4,
                stalled_seconds=900.0,
                silent_for=900.0,
                reason=EnumConsumerStallReason.STALLED_NO_ASSIGNMENT,
            ),
            _status(ready=True, backlog=0),
        )
    )
    assert _dimension(event, "consumer_sync").status == "CRITICAL"  # type: ignore[union-attr]


def test_a_failed_rejoin_is_named_in_the_detail_separately_from_the_stall() -> None:
    """The remedy failing and the group being behind are two different facts."""
    detail = describe_consumer_sync(
        (
            _status(
                ready=False,
                backlog=0,
                silent_for=400.0,
                reason=EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING,
                last_rejoin_failed=True,
            ),
        )
    )
    assert "rejoin" in detail.lower()


# --- the chain to `docker ps` ----------------------------------------------


@pytest.mark.asyncio
async def test_the_container_healthcheck_fails_on_a_wedged_group() -> None:
    """End to end: the dimension reaches the exit code AC7 reads.

    Everything between the monitor and the process exit already existed. This
    asserts the new dimension actually travels it, which is the difference
    between this ticket and a metric.
    """
    event = await _run_once(
        _FakeBus(
            _status(
                ready=False,
                backlog=16,
                stalled_seconds=3_000.0,
                silent_for=3_000.0,
                reason=EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING,
            )
        )
    )
    block = build_runtime_health_block(event)  # type: ignore[arg-type]
    payload = {
        "status": fold_runtime_verdict_into_status("healthy", event.status),  # type: ignore[attr-defined]
        "details": {"is_running": True, **block},
    }
    # The payload the runtime would actually serve, round-tripped as JSON so
    # this cannot pass on a shape the HTTP surface would not produce.
    outcome = evaluate_health_response(
        http_status=200, payload=json.loads(json.dumps(payload))
    )

    assert outcome.exit_code != 0, (
        "the container healthcheck read a wedged consumer group and exited "
        "zero -- `docker ps` would stay green and AC7 would never fire"
    )
    # `runtime_unhealthy`, not `runtime_critical`: a CRITICAL verdict is folded
    # into the body's top-level status by `fold_runtime_verdict_into_status`,
    # and the healthcheck reads that before it reaches the dimension block. The
    # exit code -- the only thing `docker ps` and AC7 see -- is the same either
    # way. Pinned rather than loosened so a future change to the fold shows up
    # here as a decision instead of passing silently.
    assert outcome.reason == "runtime_unhealthy"

    # The dimension name is NOT on that verdict's detail, because that branch
    # short-circuits before `_describe_dimensions`. It is in the /health body,
    # which is where an operator diagnosing the red container reads it, so
    # assert it there rather than pretend the healthcheck carries it.
    served = json.dumps(payload)
    assert "consumer_sync" in served
    assert "lag=16" in served and "stalled=3000s" in served


@pytest.mark.asyncio
async def test_the_container_healthcheck_passes_on_an_idle_group() -> None:
    """Positive control for the same chain."""
    event = await _run_once(_FakeBus(_status(ready=True, backlog=0, silent_for=42.0)))
    block = build_runtime_health_block(event)  # type: ignore[arg-type]
    payload = {
        "status": fold_runtime_verdict_into_status("healthy", event.status),  # type: ignore[attr-defined]
        "details": {"is_running": True, **block},
    }
    outcome = evaluate_health_response(
        http_status=200, payload=json.loads(json.dumps(payload))
    )
    assert outcome.exit_code == 0, outcome.detail


# --- the derived verdict ---------------------------------------------------


def test_readiness_is_derived_from_the_measurement_not_asserted() -> None:
    """``ready`` cannot be stated; it follows from the numbers beside it.

    Same reason ``ModelConsumerStallVerdict.should_rejoin`` and
    ``ModelLabPassReceipt``'s verdict are derived: a field an emitter can
    state wrongly is a field that will eventually be stated wrongly, and this
    one exists because three surfaces stated health wrongly for fifty minutes.
    """
    with pytest.raises(ValueError):
        ModelConsumerSyncStatus(
            topic=TOPIC,
            consumer_group=GROUP,
            evaluated=True,
            reason=EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING,
            backlog_records=16,
            seconds_since_last_record=3_000.0,
            stalled_seconds=3_000.0,
            assigned_partitions=1,
            broker_reachable=True,
            rejoin_count=0,
            last_rejoin_failed=False,
            unready_after_seconds=600.0,
            ready=True,  # type: ignore[call-arg]
        )


def test_a_stall_shorter_than_the_window_is_still_ready() -> None:
    """The window is what keeps readiness from racing the self-heal."""
    status = ModelConsumerSyncStatus(
        topic=TOPIC,
        consumer_group=GROUP,
        evaluated=True,
        reason=EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING,
        backlog_records=16,
        seconds_since_last_record=200.0,
        stalled_seconds=200.0,
        assigned_partitions=1,
        broker_reachable=True,
        rejoin_count=0,
        last_rejoin_failed=False,
        unready_after_seconds=600.0,
    )
    assert status.ready


def test_an_unreachable_broker_is_not_reported_as_an_out_of_sync_consumer() -> None:
    """A broker that is down is a broker problem.

    The rejoin path already refuses to recreate into an unreachable broker.
    Readiness makes the same call for the same reason: reporting the consumer
    as the fault would point recovery at the wrong container, which is how
    fifty minutes got spent on the runtime during the 09-17 outage.
    """
    status = ModelConsumerSyncStatus(
        topic=TOPIC,
        consumer_group=GROUP,
        evaluated=True,
        reason=EnumConsumerStallReason.NOT_STALLED_BROKER_UNREACHABLE,
        backlog_records=0,
        seconds_since_last_record=3_000.0,
        stalled_seconds=0.0,
        assigned_partitions=1,
        broker_reachable=False,
        rejoin_count=0,
        last_rejoin_failed=False,
        unready_after_seconds=600.0,
    )
    assert status.ready


def test_a_status_carries_the_window_it_was_judged_against() -> None:
    """Self-describing evidence. A reader must not need the config to read it."""
    status = _status(ready=True)
    assert status.unready_after_seconds == 600.0
