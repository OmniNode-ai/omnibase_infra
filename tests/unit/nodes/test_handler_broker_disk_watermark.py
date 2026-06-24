# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerBrokerDiskWatermark.

TDD coverage for the broker disk watermark probe (OMN-13009 ratchet).
All docker / shutil interactions are injected via seams — no real docker daemon.

Ticket: OMN-13009
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_infra.nodes.node_broker_disk_watermark_compute.handlers.handler_broker_disk_watermark import (
    HandlerBrokerDiskWatermark,
    _classify,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.enum_disk_severity import (
    EnumDiskSeverity,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.model_broker_disk_watermark_input import (
    ModelBrokerDiskWatermarkInput,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.model_broker_disk_watermark_output import (
    ModelBrokerDiskWatermarkOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GiB = 1024**3
_MiB = 1024**2

_DEMO_DAY_INFO = {"DockerRootDir": "/data"}


def _make_input(
    *,
    broker_volume_names: tuple[str, ...] = (),
    warn_threshold: float = 0.85,
    p0_threshold: float = 0.95,
    min_free_bytes_floor: int = 10 * _MiB,
    docker_info_runner=None,
    disk_usage_runner=None,
) -> ModelBrokerDiskWatermarkInput:
    return ModelBrokerDiskWatermarkInput(
        broker_volume_names=broker_volume_names,
        warn_threshold=warn_threshold,
        p0_threshold=p0_threshold,
        min_free_bytes_floor=min_free_bytes_floor,
        docker_info_runner=docker_info_runner,
        disk_usage_runner=disk_usage_runner,
    )


def _healthy_disk_usage(path: str) -> tuple[int, int, int]:
    """70% used — well below warn."""
    total = 100 * _GiB
    used = 70 * _GiB
    free = total - used
    return total, used, free


def _warn_disk_usage(path: str) -> tuple[int, int, int]:
    """88% used — between warn and p0."""
    total = 100 * _GiB
    used = int(0.88 * total)
    free = total - used
    return total, used, free


def _p0_disk_usage(path: str) -> tuple[int, int, int]:
    """96% used — above p0."""
    total = 100 * _GiB
    used = int(0.96 * total)
    free = total - used
    return total, used, free


def _demo_day_disk_usage(path: str) -> tuple[int, int, int]:
    """100% used — exact demo-day condition: 0 free bytes."""
    total = 3_600 * _GiB
    return total, total, 0


# ---------------------------------------------------------------------------
# Tests for classify helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyHelper:
    """Unit tests for the _classify pure function."""

    def test_clean(self) -> None:
        severity, below_floor = _classify(
            0.50,
            50 * _GiB,
            warn_threshold=0.85,
            p0_threshold=0.95,
            min_free_bytes_floor=10 * _MiB,
        )
        assert severity == EnumDiskSeverity.CLEAN
        assert not below_floor

    def test_warn_boundary(self) -> None:
        severity, _ = _classify(
            0.85,
            15 * _GiB,
            warn_threshold=0.85,
            p0_threshold=0.95,
            min_free_bytes_floor=10 * _MiB,
        )
        assert severity == EnumDiskSeverity.WARN

    def test_p0_boundary(self) -> None:
        severity, _ = _classify(
            0.95,
            5 * _GiB,
            warn_threshold=0.85,
            p0_threshold=0.95,
            min_free_bytes_floor=10 * _MiB,
        )
        assert severity == EnumDiskSeverity.P0

    def test_below_min_free_floor_escalates_to_p0(self) -> None:
        """Even at 50% usage, zero free bytes must be classified P0."""
        severity, below_floor = _classify(
            0.50,
            0,
            warn_threshold=0.85,
            p0_threshold=0.95,
            min_free_bytes_floor=10 * _MiB,
        )
        assert severity == EnumDiskSeverity.P0
        assert below_floor

    def test_floor_exactly_at_boundary(self) -> None:
        """free_bytes == floor is still P0 (boundary: <=)."""
        floor = 10 * _MiB
        severity, below_floor = _classify(
            0.50,
            floor,
            warn_threshold=0.85,
            p0_threshold=0.95,
            min_free_bytes_floor=floor,
        )
        assert severity == EnumDiskSeverity.P0
        assert below_floor

    def test_one_byte_above_floor_not_escalated(self) -> None:
        floor = 10 * _MiB
        severity, below_floor = _classify(
            0.50,
            floor + 1,
            warn_threshold=0.85,
            p0_threshold=0.95,
            min_free_bytes_floor=floor,
        )
        assert severity == EnumDiskSeverity.CLEAN
        assert not below_floor


# ---------------------------------------------------------------------------
# Tests for HandlerBrokerDiskWatermark
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerBrokerDiskWatermark:
    """Unit tests for the full handler."""

    # ------------------------------------------------------------------ #
    # Handler properties                                                   #
    # ------------------------------------------------------------------ #

    def test_handler_type(self) -> None:
        from omnibase_infra.enums import EnumHandlerType

        handler = HandlerBrokerDiskWatermark()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_is_compute(self) -> None:
        from omnibase_infra.enums import EnumHandlerTypeCategory

        handler = HandlerBrokerDiskWatermark()
        assert handler.handler_category == EnumHandlerTypeCategory.COMPUTE

    # ------------------------------------------------------------------ #
    # Healthy state (no volumes)                                          #
    # ------------------------------------------------------------------ #

    def test_clean_data_root_no_volumes(self) -> None:
        """Healthy filesystem → CLEAN, no p0/warn labels, docker_root_dir populated."""
        handler = HandlerBrokerDiskWatermark()
        inp = _make_input(
            docker_info_runner=lambda: _DEMO_DAY_INFO,
            disk_usage_runner=_healthy_disk_usage,
        )
        out = handler.probe_disk_watermark(inp)
        assert isinstance(out, ModelBrokerDiskWatermarkOutput)
        assert out.max_severity == EnumDiskSeverity.CLEAN
        assert out.p0_labels == ()
        assert out.warn_labels == ()
        assert out.docker_root_dir == "/data"
        assert out.probe_error == ""
        assert len(out.checks) == 1
        assert out.checks[0].label == "docker-data-root"
        assert out.checks[0].path == "/data"

    # ------------------------------------------------------------------ #
    # Warn                                                                 #
    # ------------------------------------------------------------------ #

    def test_warn_data_root(self) -> None:
        handler = HandlerBrokerDiskWatermark()
        inp = _make_input(
            docker_info_runner=lambda: _DEMO_DAY_INFO,
            disk_usage_runner=_warn_disk_usage,
        )
        out = handler.probe_disk_watermark(inp)
        assert out.max_severity == EnumDiskSeverity.WARN
        assert "docker-data-root" in out.warn_labels
        assert out.p0_labels == ()

    # ------------------------------------------------------------------ #
    # P0 — over threshold                                                 #
    # ------------------------------------------------------------------ #

    def test_p0_data_root_over_threshold(self) -> None:
        handler = HandlerBrokerDiskWatermark()
        inp = _make_input(
            docker_info_runner=lambda: _DEMO_DAY_INFO,
            disk_usage_runner=_p0_disk_usage,
        )
        out = handler.probe_disk_watermark(inp)
        assert out.max_severity == EnumDiskSeverity.P0
        assert "docker-data-root" in out.p0_labels

    # ------------------------------------------------------------------ #
    # Exact demo-day condition: 0 free bytes                              #
    # ------------------------------------------------------------------ #

    def test_demo_day_zero_free_bytes_is_p0(self) -> None:
        """0 free bytes → P0 via min_free_bytes_floor cross-check."""
        handler = HandlerBrokerDiskWatermark()
        inp = _make_input(
            docker_info_runner=lambda: _DEMO_DAY_INFO,
            disk_usage_runner=_demo_day_disk_usage,
        )
        out = handler.probe_disk_watermark(inp)
        assert out.max_severity == EnumDiskSeverity.P0
        assert out.checks[0].below_min_free_floor
        assert out.checks[0].free_bytes == 0

    # ------------------------------------------------------------------ #
    # Docker info failure → probe_error set, no checks from data-root    #
    # ------------------------------------------------------------------ #

    def test_docker_info_failure_populates_probe_error(self) -> None:
        def failing_docker_info() -> dict:
            raise RuntimeError("docker not available")

        handler = HandlerBrokerDiskWatermark()
        inp = _make_input(
            docker_info_runner=failing_docker_info,
            disk_usage_runner=_healthy_disk_usage,
        )
        out = handler.probe_disk_watermark(inp)
        assert out.probe_error != ""
        assert "docker not available" in out.probe_error
        assert out.docker_root_dir == ""
        assert len(out.checks) == 0

    # ------------------------------------------------------------------ #
    # Named broker volumes via injected disk_usage seam                  #
    # ------------------------------------------------------------------ #

    def test_broker_volumes_probed(self) -> None:
        """Named broker volume is probed when volume inspect returns mountpoint."""

        # Patch _resolve_volume_mountpoint so no real docker needed.
        def fake_disk_usage(path: str) -> tuple[int, int, int]:
            if path == "/data":
                return _healthy_disk_usage(path)
            # Simulate the broker volume path
            return _p0_disk_usage(path)

        with patch(
            "omnibase_infra.nodes.node_broker_disk_watermark_compute.handlers"
            ".handler_broker_disk_watermark._resolve_volume_mountpoint",
            return_value="/var/lib/docker/volumes/broker/_data",
        ):
            handler = HandlerBrokerDiskWatermark()
            inp = _make_input(
                broker_volume_names=("omnibase-infra-stability-test_redpanda-data",),
                docker_info_runner=lambda: _DEMO_DAY_INFO,
                disk_usage_runner=fake_disk_usage,
            )
            out = handler.probe_disk_watermark(inp)

        assert len(out.checks) == 2
        volume_check = next(
            c
            for c in out.checks
            if c.label == "omnibase-infra-stability-test_redpanda-data"
        )
        assert volume_check.severity == EnumDiskSeverity.P0
        assert out.max_severity == EnumDiskSeverity.P0

    def test_broker_volume_skipped_when_mountpoint_not_found(self) -> None:
        """Volume with no mountpoint is skipped; probe still succeeds."""
        with patch(
            "omnibase_infra.nodes.node_broker_disk_watermark_compute.handlers"
            ".handler_broker_disk_watermark._resolve_volume_mountpoint",
            return_value=None,
        ):
            handler = HandlerBrokerDiskWatermark()
            inp = _make_input(
                broker_volume_names=("missing-volume",),
                docker_info_runner=lambda: _DEMO_DAY_INFO,
                disk_usage_runner=_healthy_disk_usage,
            )
            out = handler.probe_disk_watermark(inp)

        # Only data-root check present
        assert len(out.checks) == 1
        assert out.max_severity == EnumDiskSeverity.CLEAN

    # ------------------------------------------------------------------ #
    # Max severity aggregation                                            #
    # ------------------------------------------------------------------ #

    def test_max_severity_is_worst_across_checks(self) -> None:
        """If any check is P0, max_severity is P0."""
        call_count = [0]

        def mixed_usage(path: str) -> tuple[int, int, int]:
            call_count[0] += 1
            if call_count[0] == 1:
                return _healthy_disk_usage(path)  # data-root: CLEAN
            return _p0_disk_usage(path)  # volume: P0

        with patch(
            "omnibase_infra.nodes.node_broker_disk_watermark_compute.handlers"
            ".handler_broker_disk_watermark._resolve_volume_mountpoint",
            return_value="/mnt/vol",
        ):
            handler = HandlerBrokerDiskWatermark()
            inp = _make_input(
                broker_volume_names=("broker-vol",),
                docker_info_runner=lambda: _DEMO_DAY_INFO,
                disk_usage_runner=mixed_usage,
            )
            out = handler.probe_disk_watermark(inp)

        assert out.max_severity == EnumDiskSeverity.P0

    # ------------------------------------------------------------------ #
    # Correlation ID is threaded through                                  #
    # ------------------------------------------------------------------ #

    def test_correlation_id_propagated(self) -> None:
        handler = HandlerBrokerDiskWatermark()
        inp = _make_input(
            docker_info_runner=lambda: _DEMO_DAY_INFO,
            disk_usage_runner=_healthy_disk_usage,
        )
        out = handler.probe_disk_watermark(inp)
        assert isinstance(out.correlation_id, UUID)
        assert out.correlation_id == inp.correlation_id

    # ------------------------------------------------------------------ #
    # Empty broker_volume_names (default)                                 #
    # ------------------------------------------------------------------ #

    def test_no_volumes_only_data_root(self) -> None:
        handler = HandlerBrokerDiskWatermark()
        inp = _make_input(
            docker_info_runner=lambda: _DEMO_DAY_INFO,
            disk_usage_runner=_healthy_disk_usage,
        )
        out = handler.probe_disk_watermark(inp)
        assert len(out.checks) == 1
        assert out.checks[0].label == "docker-data-root"
