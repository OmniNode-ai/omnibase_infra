# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The apply-flow dimensions driven through the REAL wiring seam (OMN-18910).

The unit tests for these two dimensions feed the process accumulator directly.
That proves the grading, and it does not prove the thing that actually failed
in OMN-18880, OMN-18905 and OMN-18769: that a projection which consumes and
writes nothing is *seen* by the surface a person reads. A counter nobody
increments grades HEALTHY forever, and an accumulator wired to nothing is
indistinguishable from a clean lane -- which is the same false all-clear the
epic exists to remove.

So this file drives the chain end to end, using the shipped dispatch callback
rather than a stand-in for it:

    _make_projection_dispatch_callback (the real wiring seam)
        -> the handler returns rows_upserted=0 (the refusal)
        -> the process apply accumulator records consumed-without-write
        -> ServiceRuntimeHealthMonitor closes the window and grades
        -> projection_apply_divergence reports DEGRADED and names the handler

The negative control is the identical path with rows landing, which must leave
both dimensions green. Without it, a dimension stuck on DEGRADED would pass the
positive case and be useless.

Related Tickets:
    - OMN-18910: these dimensions (epic OMN-18906 AC-4)
    - OMN-12245: the sibling runtime-dispatch integration coverage this follows
    - OMN-18880: the nine-hour total refusal this replays in miniature
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_projection_dispatch_callback,
)
from omnibase_infra.runtime.health.projection_apply_flow import (
    APPLY_DIVERGENCE_MIN_CONSUMED,
)
from omnibase_infra.runtime.observability.projection_apply_counters import (
    reset_projection_apply_counters_for_test,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

TOPIC = "onex.evt.omniclaude.task-delegated.v1"  # onex-topic-allow: dispatch drive
DIVERGENCE = "projection_apply_divergence"
DROPPED = "projection_delta_dropped"


@pytest.fixture(autouse=True)
def _configured_projection_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every binding DSN the shipped topology declares, as OMN-17152 requires."""
    configure_projection_dsns(monkeypatch)


@pytest.fixture(autouse=True)
def _clean_counters() -> object:
    """Each test owns the process accumulator; none inherits another's window."""
    reset_projection_apply_counters_for_test()
    yield
    reset_projection_apply_counters_for_test()


class _RefusingProjection:
    """Consumes every event and writes no row -- the OMN-18880 condition.

    It does not raise and it does not log an error, which is the whole
    difficulty: the offset commits, lag reads zero, the DLQ stays empty and
    every pre-existing dimension reports the consumer attached and moving.
    """

    topics = [TOPIC]

    def __init__(self, rows_upserted: int = 0) -> None:
        self._rows_upserted = rows_upserted
        self.calls = 0

    def handle(self, input_data: dict[str, object]) -> dict[str, int]:
        self.calls += 1
        return {"rows_upserted": self._rows_upserted}


class _SilentBus:
    """An event bus with no consumer-sync surface, as the in-memory bus has."""


def _manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.total_discovered = 0
    manifest.total_errors = 0
    manifest.errors = ()
    manifest.all_subscribe_topics.return_value = ()
    return manifest


def _envelope() -> MagicMock:
    envelope = MagicMock()
    envelope.topic = TOPIC
    envelope.payload = {"correlation_id": "corr-1", "task_type": "release-proof"}
    return envelope


def _drive(handler: _RefusingProjection, events: int) -> None:
    """Wire the handler through the shipped seam and dispatch N real events."""
    callback = _make_projection_dispatch_callback(
        handler,
        projection_database_target("delegation_events"),
        (TOPIC,),
    )
    for _ in range(events):
        asyncio.run(callback(_envelope()))


async def _run_monitor_once() -> object:
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


def _dimension(event: object, name: str) -> object:
    for dimension in event.dimensions:  # type: ignore[attr-defined]
        if dimension.name == name:
            return dimension
    raise AssertionError(
        f"{name} is absent from the rendered health event. A dimension that is "
        "not rendered cannot be read by anyone, which is the failure this "
        "ticket closes, so its absence is a failure and not a skip."
    )


@pytest.mark.integration
def test_a_refusing_projection_driven_through_the_real_seam_degrades() -> None:
    """The consumed-without-written condition reaches the health surface.

    Drives the shipped dispatch callback, not a stand-in: if the recording
    call were removed from ``handler_wiring`` this test goes red, which is the
    property the unit tests cannot have.
    """
    handler = _RefusingProjection(rows_upserted=0)
    _drive(handler, APPLY_DIVERGENCE_MIN_CONSUMED)

    assert handler.calls == APPLY_DIVERGENCE_MIN_CONSUMED, (
        "the seam did not actually dispatch; the rest of this assertion would "
        "then be measuring an empty accumulator rather than a refusal"
    )

    event = asyncio.run(_run_monitor_once())
    divergence = _dimension(event, DIVERGENCE)

    assert divergence.status == "DEGRADED"  # type: ignore[attr-defined]
    assert type(handler).__name__ in divergence.detail  # type: ignore[attr-defined]


@pytest.mark.integration
def test_negative_control_the_same_seam_with_rows_landing_stays_green() -> None:
    """Identical traffic, rows written. Both dimensions must stay HEALTHY.

    Without this the positive case above would also pass a dimension welded to
    DEGRADED, which detects nothing and gets muted within a day.
    """
    handler = _RefusingProjection(rows_upserted=1)
    _drive(handler, APPLY_DIVERGENCE_MIN_CONSUMED)

    assert handler.calls == APPLY_DIVERGENCE_MIN_CONSUMED

    event = asyncio.run(_run_monitor_once())

    assert _dimension(event, DIVERGENCE).status == "HEALTHY"  # type: ignore[attr-defined]
    assert _dimension(event, DROPPED).status == "HEALTHY"  # type: ignore[attr-defined]


@pytest.mark.integration
def test_both_dimensions_are_rendered_after_a_real_dispatch() -> None:
    """A silent evaluation and an evaluation that never ran look identical.

    So the positive control asserts presence in the rendered tree, by name,
    rather than only asserting that nothing was raised.
    """
    _drive(_RefusingProjection(rows_upserted=1), 1)

    event = asyncio.run(_run_monitor_once())
    names = {d.name for d in event.dimensions}  # type: ignore[attr-defined]

    assert {DIVERGENCE, DROPPED} <= names


@pytest.mark.integration
def test_a_projection_wired_but_never_dispatched_is_still_in_scope() -> None:
    """Registration happens at WIRING time, not on first dispatch.

    A consumer that attaches and then takes nothing is exactly the shape that
    must not vanish from the dimension that exists to notice it. Deriving
    scope from observed traffic is how that disappearance happens, so the
    handler here is wired and never driven.
    """
    handler = _RefusingProjection(rows_upserted=0)
    _make_projection_dispatch_callback(
        handler,
        projection_database_target("delegation_events"),
        (TOPIC,),
    )

    assert handler.calls == 0

    event = asyncio.run(_run_monitor_once())
    divergence = _dimension(event, DIVERGENCE)

    # In scope and reported, and NOT degraded: it consumed nothing, so it has
    # not diverged. The fact under test is that it is counted at all.
    assert divergence.status == "HEALTHY"  # type: ignore[attr-defined]
    assert "1 dispatching projection(s)" in divergence.detail or "0 consumed" in (  # type: ignore[attr-defined]
        divergence.detail  # type: ignore[attr-defined]
    )
