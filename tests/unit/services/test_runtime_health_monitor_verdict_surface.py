# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the monitor's published verdict surface (OMN-15217).

Covers the two changes that make the runtime's semantic health readable outside
the container's log stream:

1. ``latest_event`` retention — the verdict a health endpoint can serve.
2. ``discovery_errors`` details that name the failing entry points instead of
   reporting a bare count.

Related Tickets:
    - OMN-15217: stability-lane runtime reports DEGRADED while Docker reads healthy
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
    _describe_discovery_errors,
)


def _manifest(*, contracts: int = 296, errors: tuple[object, ...] = ()) -> MagicMock:
    manifest = MagicMock()
    manifest.total_discovered = contracts
    manifest.total_errors = len(errors)
    manifest.errors = errors
    manifest.all_subscribe_topics.return_value = ()
    return manifest


def _error(entry_point_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        entry_point_name=entry_point_name,
        package_name="omnibase_infra",
        error="ModuleNotFoundError: No module named 'omnimemory.utils'",
    )


@pytest.mark.unit
class TestDescribeDiscoveryErrors:
    """The detail string is the only durable record of *which* contracts failed.

    The boot-time ``Failed to load entry point ...`` lines roll out of the
    container log buffer long before anyone investigates — OMN-15217 had to
    reconstruct them from a live container that happened to still be up.
    """

    def test_failing_entry_points_are_named(self) -> None:
        detail = _describe_discovery_errors(
            _manifest(errors=(_error("node_alpha"), _error("node_beta"))), 2
        )
        assert detail == "2 contract(s) failed to load: node_alpha, node_beta"

    def test_long_lists_are_capped_with_a_remainder(self) -> None:
        errors = tuple(_error(f"node_{index}") for index in range(12))
        detail = _describe_discovery_errors(_manifest(errors=errors), 12)

        assert detail.startswith("12 contract(s) failed to load: node_0")
        assert "+4 more" in detail
        assert "node_8" not in detail

    def test_manifest_without_error_details_degrades_to_the_count(self) -> None:
        """ProtocolAutoWiringManifestLike only guarantees the counts."""
        manifest = MagicMock()
        manifest.total_errors = 4
        # A MagicMock attribute is not a Sequence — the defensive read must not
        # raise inside a health check.
        detail = _describe_discovery_errors(manifest, 4)
        assert detail == "4 contract(s) failed to load"

    def test_unnamed_errors_degrade_to_the_count(self) -> None:
        detail = _describe_discovery_errors(
            _manifest(errors=(SimpleNamespace(error="boom"),)), 1
        )
        assert detail == "1 contract(s) failed to load"


@pytest.mark.unit
class TestLatestEventRetention:
    """The verdict must outlive the log line that reports it."""

    @pytest.mark.asyncio
    async def test_latest_event_is_none_before_the_first_cycle(self) -> None:
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="")
        assert monitor.latest_event is None

    @pytest.mark.asyncio
    async def test_run_once_publishes_a_degraded_verdict(self) -> None:
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="")

        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=_manifest(errors=(_error("node_alpha"), _error("node_beta"))),
        ):
            event = await monitor.run_once()

        assert event.status == "DEGRADED"
        assert monitor.latest_event is event
        assert monitor.latest_event is not None
        assert monitor.latest_event.discovery_error_count == 2

        discovery = next(
            dimension
            for dimension in monitor.latest_event.dimensions
            if dimension.name == "discovery_errors"
        )
        assert discovery.status == "DEGRADED"
        assert "node_alpha" in discovery.detail

    @pytest.mark.asyncio
    async def test_latest_event_tracks_the_most_recent_cycle(self) -> None:
        monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="")

        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=_manifest(errors=(_error("node_alpha"),)),
        ):
            first = await monitor.run_once()
        with patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=_manifest(),
        ):
            second = await monitor.run_once()

        assert first.status == "DEGRADED"
        assert second.status == "HEALTHY"
        assert monitor.latest_event is second
