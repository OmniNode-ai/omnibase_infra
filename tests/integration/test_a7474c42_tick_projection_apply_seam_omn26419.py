# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev job a7474c42 driven through the REAL wiring seam and the health monitor (OMN-26419).

The unit file beside the recorded fixtures replays the log through
``_record_projection_apply``. That proves the grading of a tick, and it does not
prove the thing that rolled dev back: that a projection subscribed to the
runtime tick, dispatched through the shipped callback, reads HEALTHY on the
surface ``ServiceRuntimeHealthMonitor`` renders.

    _make_projection_dispatch_callback (the real wiring seam)
        -> a tick envelope; the handler sweeps nothing and returns rows_upserted=0
        -> the process apply accumulator registers the projection, counts no event
        -> ServiceRuntimeHealthMonitor closes the window and grades
        -> projection_apply_divergence reads HEALTHY

The positive control is the identical path on the recorded source topic, which
must still read DEGRADED when it consumes and writes nothing. Without it a
dimension that ignored everything would pass the tick case.

Related Tickets:
    - OMN-26419: this regression (dev job a7474c42)
    - OMN-18910: the apply-flow dimensions this drives, unchanged here
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.models.health.model_runtime_health_check_event import (
    ModelRuntimeHealthCheckEvent,
)
from omnibase_infra.models.health.model_runtime_health_dimension import (
    ModelRuntimeHealthDimension,
)
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
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_RUNTIME_TICK
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "dev_job_a7474c42"
DIVERGENCE = "projection_apply_divergence"
_TOPIC = re.compile(r"topic=(?P<topic>\S+)")


def _recorded_source_topic() -> str:
    """The non-tick topic the recorded writer consumed in the a7474c42 excerpt."""
    topics = {
        m["topic"]
        for line in (FIXTURES / "container_log_excerpt.txt").read_text().splitlines()
        if (m := _TOPIC.search(line))
    }
    (source_topic,) = topics - {SUFFIX_RUNTIME_TICK}
    return source_topic


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


class EvalResultsProjectionWriter:
    """Stands in for the recorded handler: it writes no row on any dispatch.

    The tick finds no deadline to sweep, and the source dispatches here are the
    refusal shape. The seam keys the projection on this class name.
    """

    def __init__(self, topics: tuple[str, ...]) -> None:
        self.topics = list(topics)
        self.calls = 0

    def handle(self, input_data: dict[str, object]) -> dict[str, int]:
        self.calls += 1
        return {"rows_upserted": 0}


def _manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.total_discovered = 0
    manifest.total_errors = 0
    manifest.errors = ()
    manifest.all_subscribe_topics.return_value = ()
    return manifest


def _envelope(topic: str) -> MagicMock:
    envelope = MagicMock()
    envelope.topic = topic
    envelope.payload = {"correlation_id": "corr-a7474c42"}
    return envelope


def _drive(topic: str, events: int) -> EvalResultsProjectionWriter:
    """Wire the handler through the shipped seam and dispatch N real events."""
    handler = EvalResultsProjectionWriter(
        (SUFFIX_RUNTIME_TICK, _recorded_source_topic())
    )
    callback = _make_projection_dispatch_callback(
        handler,
        projection_database_target("delegation_events"),
        tuple(handler.topics),
    )
    for _ in range(events):
        asyncio.run(callback(_envelope(topic)))
    assert handler.calls == events, "the seam did not dispatch every event"
    return handler


async def _run_monitor_once() -> ModelRuntimeHealthCheckEvent:
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=None,
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


def _divergence(event: ModelRuntimeHealthCheckEvent) -> ModelRuntimeHealthDimension:
    for dimension in event.dimensions:
        if dimension.name == DIVERGENCE:
            return dimension
    raise AssertionError(f"{DIVERGENCE} is absent from the rendered health event")


@pytest.mark.integration
def test_a7474c42_ticks_through_the_real_seam_read_healthy() -> None:
    """A writer that only receives quiet ticks is not a divergence."""
    _drive(SUFFIX_RUNTIME_TICK, APPLY_DIVERGENCE_MIN_CONSUMED * 5)

    event = asyncio.run(_run_monitor_once())

    assert _divergence(event).status == "HEALTHY"


@pytest.mark.integration
def test_a7474c42_source_events_through_the_real_seam_still_degrade() -> None:
    """Positive control: a real source topic that writes nothing still grades."""
    handler = _drive(_recorded_source_topic(), APPLY_DIVERGENCE_MIN_CONSUMED)

    event = asyncio.run(_run_monitor_once())
    divergence = _divergence(event)

    assert divergence.status == "DEGRADED"
    assert type(handler).__name__ in divergence.detail
