# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``dispatch_deadline`` dimension (OMN-19355).

Once the consume loop abandons a hung dispatch and keeps polling, the group
stays in sync, so ``consumer_sync`` stays green. The parked handler is then
visible only here: DEGRADED from the first abandoned dispatch, CRITICAL at the
bus's orphan limit, which is the grade that fails the container healthcheck on
every lane regardless of its ``--degraded-policy``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.models.health.model_dispatch_deadline_status import (
    ModelDispatchDeadlineStatus,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    DISPATCH_DEADLINE_UNAVAILABLE,
    ServiceRuntimeHealthMonitor,
    evaluate_dispatch_deadline,
)

pytestmark = pytest.mark.unit

ORPHAN = (
    "onex.evt.omnibase-infra.lab-lane-health.v1 partition=0 offset=51808 "
    "subscription=sub-1 age=612s"
)


def _status(orphaned: int, *, limit: int = 3) -> ModelDispatchDeadlineStatus:
    return ModelDispatchDeadlineStatus(
        deadline_seconds=600.0,
        orphan_limit=limit,
        orphaned_dispatches=orphaned,
        deadline_expiries_total=orphaned,
        oldest_orphan_age_seconds=612.0 if orphaned else 0.0,
        orphans=tuple(ORPHAN for _ in range(orphaned)),
    )


class _FakeBus:
    def __init__(self, status: ModelDispatchDeadlineStatus) -> None:
        self._status = status

    def dispatch_deadline_status(self) -> ModelDispatchDeadlineStatus:
        return self._status


class _SilentBus:
    """A bus with no consume loop (the in-memory bus)."""


def test_no_abandoned_dispatch_is_healthy_and_says_what_it_measured() -> None:
    status, detail = evaluate_dispatch_deadline(_FakeBus(_status(0)))
    assert status == "HEALTHY"
    assert "0 abandoned" in detail
    assert "600s" in detail


def test_one_abandoned_dispatch_is_degraded_and_named() -> None:
    status, detail = evaluate_dispatch_deadline(_FakeBus(_status(1)))
    assert status == "DEGRADED"
    assert "offset=51808" in detail, "the parked record must be named"


def test_the_orphan_limit_is_critical() -> None:
    status, _ = evaluate_dispatch_deadline(_FakeBus(_status(3)))
    assert status == "CRITICAL"


def test_a_bus_that_cannot_answer_is_healthy_and_says_so() -> None:
    status, detail = evaluate_dispatch_deadline(_SilentBus())
    assert status == "HEALTHY"
    assert detail == DISPATCH_DEADLINE_UNAVAILABLE


class _BrokenBus:
    def dispatch_deadline_status(self) -> ModelDispatchDeadlineStatus:
        raise RuntimeError("boom")


def test_a_read_failure_is_degraded_not_silent() -> None:
    status, detail = evaluate_dispatch_deadline(_BrokenBus())
    assert status == "DEGRADED"
    assert "boom" in detail


@pytest.mark.asyncio
async def test_the_monitor_publishes_the_dimension_and_aggregates_it() -> None:
    """The dimension is not only computed: it reaches the verdict."""
    manifest = MagicMock()
    manifest.total_discovered = 0
    manifest.total_errors = 0
    manifest.errors = ()
    manifest.all_subscribe_topics.return_value = ()
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=_FakeBus(_status(3)),  # type: ignore[arg-type]
        bootstrap_servers="",
        check_interval_seconds=300.0,
        boot_grace_seconds=0.0,
    )
    with (
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ),
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._filter_manifest_for_runtime_profile",
            side_effect=lambda m: m,
        ),
    ):
        event = await monitor.run_once()
    dimension = next(
        (d for d in event.dimensions if d.name == "dispatch_deadline"),  # type: ignore[union-attr]
        None,
    )
    assert dimension is not None
    assert dimension.status == "CRITICAL"
    assert event.status == "CRITICAL"  # type: ignore[union-attr]
