# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the runtime health verdict projection (OMN-15217).

Related Tickets:
    - OMN-15217: stability-lane runtime reports DEGRADED while Docker reads healthy
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.enum_runtime_readiness_state import (
    EnumRuntimeReadinessState,
)
from omnibase_infra.models.health.model_runtime_health_check_event import (
    ModelRuntimeHealthCheckEvent,
)
from omnibase_infra.models.health.model_runtime_health_dimension import (
    ModelRuntimeHealthDimension,
)
from omnibase_infra.runtime.health.runtime_health_block import (
    build_runtime_health_block,
    fold_attach_readiness_into_status,
    fold_runtime_verdict_into_status,
)

_OBSERVED_AT = datetime(2026, 7, 27, 12, 58, 21, tzinfo=UTC)


def _event(
    status: str = "DEGRADED", *, timestamp: datetime | None = None
) -> ModelRuntimeHealthCheckEvent:
    return ModelRuntimeHealthCheckEvent(
        correlation_id=uuid4(),
        timestamp=timestamp or _OBSERVED_AT,
        status=status,  # type: ignore[arg-type]
        dimensions=(
            ModelRuntimeHealthDimension(
                name="discovery_errors",
                status="DEGRADED",
                detail="4 contract(s) failed to load: node_alpha",
            ),
            ModelRuntimeHealthDimension(
                name="consumer_coverage",
                status="HEALTHY",
                detail="All 643 expected consumer group(s) covered",
            ),
        ),
        contract_count=296,
        discovery_error_count=4,
        consumer_group_count=643,
    )


@pytest.mark.unit
class TestBuildRuntimeHealthBlock:
    def test_absent_event_renders_no_block(self) -> None:
        """No verdict is a distinct state from a healthy verdict."""
        assert build_runtime_health_block(None) is None

    def test_block_carries_status_counts_and_dimensions(self) -> None:
        block = build_runtime_health_block(
            _event(), now=_OBSERVED_AT + timedelta(seconds=42)
        )

        assert block is not None
        assert block["status"] == "DEGRADED"
        assert block["contract_count"] == 296
        assert block["discovery_error_count"] == 4
        assert block["age_seconds"] == 42.0
        assert block["observed_at"] == "2026-07-27T12:58:21+00:00"
        assert block["dimensions"] == [
            {
                "name": "discovery_errors",
                "status": "DEGRADED",
                "detail": "4 contract(s) failed to load: node_alpha",
            },
            {
                "name": "consumer_coverage",
                "status": "HEALTHY",
                "detail": "All 643 expected consumer group(s) covered",
            },
        ]

    def test_naive_timestamp_is_treated_as_utc(self) -> None:
        block = build_runtime_health_block(
            _event(timestamp=_OBSERVED_AT.replace(tzinfo=None)),
            now=_OBSERVED_AT + timedelta(seconds=10),
        )
        assert block is not None
        assert block["age_seconds"] == 10.0

    def test_clock_skew_never_yields_a_negative_age(self) -> None:
        block = build_runtime_health_block(
            _event(), now=_OBSERVED_AT - timedelta(seconds=30)
        )
        assert block is not None
        assert block["age_seconds"] == 0.0


@pytest.mark.unit
class TestFoldRuntimeVerdictIntoStatus:
    @pytest.mark.parametrize(
        ("payload_status", "verdict_status", "expected"),
        [
            ("healthy", None, "healthy"),
            ("healthy", "HEALTHY", "healthy"),
            ("healthy", "DEGRADED", "degraded"),
            ("healthy", "CRITICAL", "unhealthy"),
            ("degraded", "HEALTHY", "degraded"),
            ("degraded", "CRITICAL", "unhealthy"),
            ("unhealthy", "HEALTHY", "unhealthy"),
            ("unhealthy", None, "unhealthy"),
        ],
    )
    def test_reported_status_is_the_worse_of_the_two(
        self, payload_status: str, verdict_status: str | None, expected: str
    ) -> None:
        assert (
            fold_runtime_verdict_into_status(payload_status, verdict_status)  # type: ignore[arg-type]
            == expected
        )

    def test_a_live_process_with_a_degraded_runtime_is_not_healthy(self) -> None:
        """The exact stability-lane case: process fine, runtime degraded.

        RuntimeHostProcess.health_check() only sees process-local state, so it
        reported healthy while four contracts were failing to load.
        """
        assert fold_runtime_verdict_into_status("healthy", "DEGRADED") == "degraded"


@pytest.mark.unit
class TestFoldAttachReadinessIntoStatus:
    """OMN-15642: mirrors TestFoldRuntimeVerdictIntoStatus for boot attach readiness.

    Before OMN-15642, ``ModelRuntimeAttachReadiness`` (OMN-15512) reached only
    ``details.components.runtime_wiring`` -- a nested payload detail nothing
    upstream reads -- while the top-level ``status`` field stayed "healthy"
    with HTTP 200 even when a boot-wired Kafka consumer contract silently
    failed to attach its subscription. This is the exact class of gap
    OMN-15217 already closed for the runtime-health-monitor verdict above;
    ``fold_attach_readiness_into_status`` closes the SAME gap for the
    attach-readiness aggregate.
    """

    @pytest.mark.parametrize(
        ("payload_status", "readiness_state", "expected"),
        [
            ("healthy", None, "healthy"),
            ("healthy", EnumRuntimeReadinessState.READY, "healthy"),
            ("healthy", EnumRuntimeReadinessState.DEGRADED, "degraded"),
            ("healthy", EnumRuntimeReadinessState.FAILED, "unhealthy"),
            ("degraded", EnumRuntimeReadinessState.READY, "degraded"),
            ("degraded", EnumRuntimeReadinessState.FAILED, "unhealthy"),
            ("unhealthy", EnumRuntimeReadinessState.READY, "unhealthy"),
            ("unhealthy", None, "unhealthy"),
        ],
    )
    def test_reported_status_is_the_worse_of_the_two(
        self,
        payload_status: str,
        readiness_state: EnumRuntimeReadinessState | None,
        expected: str,
    ) -> None:
        assert (
            fold_attach_readiness_into_status(payload_status, readiness_state)  # type: ignore[arg-type]
            == expected
        )

    def test_a_live_boot_with_a_dropped_consumer_is_not_reported_healthy(self) -> None:
        """OMN-15642 live incident: boot green, one consumer silently absent.

        onex-dev run 30720296789 (and the n=2 reproduction, run 30721670043):
        deploy-onex-staging steps 30-36 (rollout, digest triple-match,
        dashboard health, post-deploy verification, staleness) all passed
        while the correlation-keyed delegation projection and the unified
        system-event stream stayed silently empty -- surfacing only ~60s
        later at the terminal business-proof gate. Pre-fix, /health's status
        field could not have shown this even if a gate had checked it: the
        DEGRADED attach_readiness aggregate never reached `status`.
        """
        assert (
            fold_attach_readiness_into_status(
                "healthy", EnumRuntimeReadinessState.DEGRADED
            )
            == "degraded"
        )
