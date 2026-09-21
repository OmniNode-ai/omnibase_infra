# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two apply-flow dimensions, walked to the surfaces a person reads.

Computing a verdict is not the deliverable. The verdict has to reach the health
event, the ``/health`` body and the container healthcheck, because a dimension
folded into nothing is exactly the shape of the three green surfaces this
ticket exists to close. So these tests walk the chain rather than its first
link:

    apply counters record  ->  window closes  ->  monitor renders two dimensions
        ->  /health body carries them  ->  container healthcheck exits non-zero

Where the failure is delivered to a person: the DEGRADED dimension is logged as
a warning by the monitor's own cycle, served on ``/health``, shipped on the
runtime health event, and folded into the container healthcheck's exit code,
which is what makes ``docker ps`` say ``unhealthy`` — the one fact the deploy
agent's force-recreate backstop probes.

Related Tickets:
    - OMN-18910: these dimensions (epic OMN-18906 AC-4)
    - OMN-15217: the verdict -> ``/health`` fold they ride
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.health.container_healthcheck import (
    evaluate_health_response,
)
from omnibase_infra.runtime.health.projection_apply_flow import (
    APPLY_DIVERGENCE_MIN_CONSUMED,
)
from omnibase_infra.runtime.health.runtime_health_block import (
    build_runtime_health_block,
    fold_runtime_verdict_into_status,
)
from omnibase_infra.runtime.observability.projection_apply_counters import (
    ProjectionApplyCounters,
    get_projection_apply_counters,
    reset_projection_apply_counters_for_test,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
)

pytestmark = pytest.mark.unit

PROJECTION = "FleetLivenessProjectionWriter"
TOPIC = "onex.evt.runner-fleet.v1"  # onex-topic-allow: replay of a recorded incident

DIVERGENCE = "projection_apply_divergence"
DROPPED = "projection_delta_dropped"


@pytest.fixture(autouse=True)
def _clean_counters() -> object:
    """Each test owns the process accumulator; none inherits another's window."""
    reset_projection_apply_counters_for_test()
    yield
    reset_projection_apply_counters_for_test()


class _SilentBus:
    """An event bus with no consumer-sync surface, as the in-memory bus has."""


def _manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.total_discovered = 0
    manifest.total_errors = 0
    manifest.errors = ()
    manifest.all_subscribe_topics.return_value = ()
    return manifest


async def _run_once() -> object:
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=_SilentBus(),  # type: ignore[arg-type]
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


def _drive_refusing_projection(windows: int = 2) -> None:
    """Consume past the floor and write nothing, over N closed windows."""
    counters = get_projection_apply_counters()
    counters.register(PROJECTION, TOPIC)
    for _ in range(windows):
        for _ in range(APPLY_DIVERGENCE_MIN_CONSUMED):
            counters.record_apply(PROJECTION, TOPIC, consumed=1, upserted=0)
        counters.close_window()


def _drive_writing_projection(windows: int = 2) -> None:
    """The same volume, with rows landing. The negative control."""
    counters = get_projection_apply_counters()
    counters.register(PROJECTION, TOPIC)
    for _ in range(windows):
        for _ in range(APPLY_DIVERGENCE_MIN_CONSUMED):
            counters.record_apply(PROJECTION, TOPIC, consumed=1, upserted=1)
        counters.close_window()


# --- the dimensions reach the event ---------------------------------------


@pytest.mark.asyncio
async def test_a_refusing_projection_degrades_the_runtime() -> None:
    """The OMN-18880 condition, end to end through the monitor."""
    _drive_refusing_projection()
    event = await _run_once()

    dimension = _dimension(event, DIVERGENCE)
    assert dimension is not None, (
        "the runtime publishes no apply-divergence dimension at all -- the "
        "nine-hour total refusal of OMN-18880 would be invisible here exactly "
        "as it was then"
    )
    assert dimension.status == "DEGRADED"  # type: ignore[attr-defined]
    assert PROJECTION in dimension.detail  # type: ignore[attr-defined]
    assert event.status in {"DEGRADED", "CRITICAL"}  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_negative_control_a_writing_projection_leaves_both_green() -> None:
    """The defect's own controls.

    Same projection, same volume, same windows, rows landing. If this went
    DEGRADED too the dimension would be measuring traffic rather than writes,
    and would be switched off inside a week.
    """
    _drive_writing_projection()
    event = await _run_once()

    divergence = _dimension(event, DIVERGENCE)
    dropped = _dimension(event, DROPPED)
    assert divergence is not None and dropped is not None
    assert divergence.status == "HEALTHY"  # type: ignore[attr-defined]
    assert dropped.status == "HEALTHY"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_both_dimensions_are_present_on_a_healthy_lane() -> None:
    """AC4: a silent evaluation and one that never ran look the same otherwise."""
    _drive_writing_projection()
    event = await _run_once()

    names = {d.name for d in event.dimensions}  # type: ignore[attr-defined]
    assert DIVERGENCE in names
    assert DROPPED in names


@pytest.mark.asyncio
async def test_a_process_with_no_projection_dispatch_is_not_degraded() -> None:
    """Most runtimes wire no projection; they must not sit permanently red."""
    event = await _run_once()

    divergence = _dimension(event, DIVERGENCE)
    dropped = _dimension(event, DROPPED)
    assert divergence is not None and dropped is not None
    assert divergence.status == "HEALTHY"  # type: ignore[attr-defined]
    assert dropped.status == "HEALTHY"  # type: ignore[attr-defined]


# --- where the failure is delivered ---------------------------------------


@pytest.mark.asyncio
async def test_the_verdict_reaches_the_health_body_and_the_container_probe() -> None:
    """A dimension nobody is served is a dimension that did not fire.

    This is the link the epic is about. The monitor's verdict has to fold into
    the ``/health`` body and out through the container healthcheck's exit
    code, because that exit code is what the deploy agent probes.
    """
    _drive_refusing_projection()
    event = await _run_once()

    block = build_runtime_health_block(event)  # type: ignore[arg-type]
    assert block is not None
    payload = {
        "status": fold_runtime_verdict_into_status("healthy", event.status),  # type: ignore[attr-defined]
        "details": {"is_running": True, **block},
    }
    # Round-tripped as JSON so this cannot pass on a shape the HTTP surface
    # would never produce.
    rendered = json.loads(json.dumps(payload))

    dimension_names = {d["name"] for d in rendered["details"]["dimensions"]}
    assert DIVERGENCE in dimension_names, (
        "the dimension is computed and then folded into nothing, which is the "
        "exact shape of the green surfaces this epic exists to close"
    )
    assert DROPPED in dimension_names

    outcome = evaluate_health_response(http_status=200, payload=rendered)
    assert outcome.exit_code != 0, (
        "a DEGRADED apply-divergence dimension must reach the container probe; "
        "otherwise docker ps still says healthy over a projection writing "
        "nothing, which is the exact surface OMN-18880 hid behind"
    )


# --- the accumulator itself ------------------------------------------------


def test_the_drop_gauge_survives_a_window_close() -> None:
    """A gauge reset every window can never show accumulation."""
    counters = ProjectionApplyCounters()
    counters.register(PROJECTION, TOPIC)
    counters.record_drop_total(PROJECTION, TOPIC, 41)
    counters.close_window()
    second = counters.close_window()

    assert second[0].deltas_dropped_total == 41
    assert second[0].consumed == 0, "the per-window counts DO reset"


def test_a_registered_projection_that_took_nothing_still_emits_a_row() -> None:
    """Alive-and-took-nothing is a fact; not-observed is a different one."""
    counters = ProjectionApplyCounters()
    counters.register(PROJECTION, TOPIC)
    window = counters.close_window()

    assert len(window) == 1
    assert window[0].projection == PROJECTION
    assert window[0].consumed == 0


def test_the_retained_ring_is_oldest_first() -> None:
    """Order is load-bearing: the drop dimension reads monotonicity across it."""
    counters = ProjectionApplyCounters()
    counters.register(PROJECTION, TOPIC)
    for total in (1, 2, 3):
        counters.record_drop_total(PROJECTION, TOPIC, total)
        counters.close_window()

    series = [window[0].deltas_dropped_total for window in counters.retained_windows()]
    assert series == [1, 2, 3]


# --- the consumer-first drop gauge -----------------------------------------


def test_a_writer_that_reports_no_gauge_is_read_exactly_as_today() -> None:
    """Consumer-first: the absence of the field changes nothing.

    This is the half OMN-18918 got wrong the expensive way and OMN-18992 then
    got right — an additive field is only additive if the consumer tolerates
    its absence, and the caller probes rather than assuming.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _extract_deltas_dropped_total,
    )

    assert _extract_deltas_dropped_total({"rows_upserted": 1}) is None
    assert _extract_deltas_dropped_total({"deltas_dropped_total": 0}) == 0, (
        "a reported zero is a writer saying it discarded nothing, which is not "
        "the same fact as a writer that does not report"
    )
    assert _extract_deltas_dropped_total({"deltas_dropped_total": 12}) == 12
    assert _extract_deltas_dropped_total({"deltas_dropped_total": "nonsense"}) is None
    assert _extract_deltas_dropped_total({"deltas_dropped_total": -3}) is None
    assert _extract_deltas_dropped_total(None) is None


def test_an_unreported_gauge_leaves_the_drop_dimension_green() -> None:
    """No producer yet must read as no finding, never as a finding."""
    counters = ProjectionApplyCounters()
    counters.register(PROJECTION, TOPIC)
    for _ in range(4):
        counters.record_apply(PROJECTION, TOPIC, consumed=5, upserted=5)
        counters.close_window()

    series = [w[0].deltas_dropped_total for w in counters.retained_windows()]
    assert series == [0, 0, 0, 0]
