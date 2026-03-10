# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Integration tests for binding resolution metrics emission.

Verifies that the OperationBindingResolver emits metrics during binding
resolution, including:

- Resolution time histogram population
- Per-operation resolution counts
- Error counting on resolution failures
- Metrics accumulation across multiple resolutions
- Metrics reset behavior

Related:
    - OMN-1644: Observability for operation bindings
    - OMN-1518: Declarative operation bindings (parent feature)
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.errors import BindingResolutionError
from omnibase_infra.models.bindings import (
    ModelOperationBindingsSubcontract,
    ModelParsedBinding,
)
from omnibase_infra.runtime.binding_resolver import OperationBindingResolver

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def resolver() -> OperationBindingResolver:
    """Create a fresh resolver for each test."""
    return OperationBindingResolver()


@pytest.fixture
def simple_bindings() -> ModelOperationBindingsSubcontract:
    """Create simple bindings with payload and context sources."""
    return ModelOperationBindingsSubcontract(
        global_bindings=[
            ModelParsedBinding(
                parameter_name="correlation_id",
                source="envelope",
                path_segments=("correlation_id",),
                required=False,
                original_expression="${envelope.correlation_id}",
            ),
        ],
        bindings={
            "db.query": [
                ModelParsedBinding(
                    parameter_name="sql",
                    source="payload",
                    path_segments=("sql",),
                    required=True,
                    original_expression="${payload.sql}",
                ),
                ModelParsedBinding(
                    parameter_name="timestamp",
                    source="context",
                    path_segments=("now_iso",),
                    required=False,
                    original_expression="${context.now_iso}",
                ),
            ],
        },
    )


@pytest.fixture
def required_missing_bindings() -> ModelOperationBindingsSubcontract:
    """Create bindings that will fail due to missing required field."""
    return ModelOperationBindingsSubcontract(
        bindings={
            "db.query": [
                ModelParsedBinding(
                    parameter_name="missing_field",
                    source="payload",
                    path_segments=("nonexistent",),
                    required=True,
                    original_expression="${payload.nonexistent}",
                ),
            ],
        },
    )


# =============================================================================
# Tests
# =============================================================================


class TestBindingResolutionMetricsEmission:
    """Integration tests verifying metrics are emitted during binding resolution."""

    def test_metrics_emitted_on_successful_resolution(
        self,
        resolver: OperationBindingResolver,
        simple_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Metrics are recorded for successful binding resolution."""
        # Initial state
        assert resolver.metrics.total_resolutions == 0

        # Resolve bindings
        correlation_id = uuid4()
        envelope = {
            "correlation_id": correlation_id,
            "payload": {"sql": "SELECT 1"},
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        result = resolver.resolve(
            operation="db.query",
            bindings_subcontract=simple_bindings,
            envelope=envelope,
            context=context,
            correlation_id=correlation_id,
        )

        # Verify resolution succeeded
        assert result.success is True

        # Verify metrics were emitted
        m = resolver.metrics
        assert m.total_resolutions == 1, (
            "total_resolutions should be 1 after one resolution"
        )
        assert m.successful_resolutions == 1, "successful_resolutions should be 1"
        assert m.failed_resolutions == 0, "failed_resolutions should be 0 after success"
        assert m.bindings_resolved_count > 0, (
            "bindings_resolved_count should reflect resolved parameters"
        )

    def test_latency_histogram_populated(
        self,
        resolver: OperationBindingResolver,
        simple_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Latency histogram buckets are populated after resolution."""
        envelope = {
            "correlation_id": uuid4(),
            "payload": {"sql": "SELECT 1"},
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        resolver.resolve(
            operation="db.query",
            bindings_subcontract=simple_bindings,
            envelope=envelope,
            context=context,
        )

        m = resolver.metrics
        # At least one histogram bucket should have a count
        total_in_histogram = sum(m.latency_histogram.values())
        assert total_in_histogram == 1, (
            f"Exactly one histogram entry expected, got {total_in_histogram}"
        )

        # Latency should be recorded
        assert m.total_latency_ms > 0, "total_latency_ms should be positive"
        assert m.min_latency_ms is not None, "min_latency_ms should be set"
        assert m.max_latency_ms is not None, "max_latency_ms should be set"

    def test_per_operation_metrics_tracked(
        self,
        resolver: OperationBindingResolver,
        simple_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Per-operation resolution counts are tracked."""
        envelope = {
            "correlation_id": uuid4(),
            "payload": {"sql": "SELECT 1"},
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        resolver.resolve(
            operation="db.query",
            bindings_subcontract=simple_bindings,
            envelope=envelope,
            context=context,
        )

        m = resolver.metrics
        assert "db.query" in m.per_operation_resolutions, (
            "per_operation_resolutions should track 'db.query'"
        )
        assert m.per_operation_resolutions["db.query"] == 1

    def test_metrics_accumulate_across_resolutions(
        self,
        resolver: OperationBindingResolver,
        simple_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Metrics accumulate across multiple resolution attempts."""
        envelope = {
            "correlation_id": uuid4(),
            "payload": {"sql": "SELECT 1"},
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        # Resolve three times
        for _ in range(3):
            resolver.resolve(
                operation="db.query",
                bindings_subcontract=simple_bindings,
                envelope=envelope,
                context=context,
            )

        m = resolver.metrics
        assert m.total_resolutions == 3
        assert m.successful_resolutions == 3
        assert m.per_operation_resolutions["db.query"] == 3

    def test_failure_metrics_on_required_binding_missing(
        self,
        resolver: OperationBindingResolver,
        required_missing_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Failed resolutions increment failure counters."""
        envelope = {
            "correlation_id": uuid4(),
            "payload": {"sql": "SELECT 1"},  # No 'nonexistent' field
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        with pytest.raises(BindingResolutionError):
            resolver.resolve(
                operation="db.query",
                bindings_subcontract=required_missing_bindings,
                envelope=envelope,
                context=context,
            )

        m = resolver.metrics
        assert m.total_resolutions == 1, (
            "total_resolutions should count failed attempts"
        )
        assert m.failed_resolutions == 1, "failed_resolutions should be 1"
        assert m.successful_resolutions == 0
        assert "db.query" in m.per_operation_errors, (
            "per_operation_errors should track the failed operation"
        )
        assert m.per_operation_errors["db.query"] == 1

    def test_mixed_success_and_failure_metrics(
        self,
        resolver: OperationBindingResolver,
        simple_bindings: ModelOperationBindingsSubcontract,
        required_missing_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Metrics correctly track interleaved successes and failures."""
        envelope_good = {
            "correlation_id": uuid4(),
            "payload": {"sql": "SELECT 1"},
        }
        envelope_bad = {
            "correlation_id": uuid4(),
            "payload": {"other": "data"},
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        # Two successes
        resolver.resolve(
            operation="db.query",
            bindings_subcontract=simple_bindings,
            envelope=envelope_good,
            context=context,
        )
        resolver.resolve(
            operation="db.query",
            bindings_subcontract=simple_bindings,
            envelope=envelope_good,
            context=context,
        )

        # One failure
        with pytest.raises(BindingResolutionError):
            resolver.resolve(
                operation="db.query",
                bindings_subcontract=required_missing_bindings,
                envelope=envelope_bad,
                context=context,
            )

        m = resolver.metrics
        assert m.total_resolutions == 3
        assert m.successful_resolutions == 2
        assert m.failed_resolutions == 1
        assert m.success_rate == pytest.approx(2 / 3)
        assert m.error_rate == pytest.approx(1 / 3)

    def test_metrics_reset(
        self,
        resolver: OperationBindingResolver,
        simple_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """reset_metrics() clears all counters to zero."""
        envelope = {
            "correlation_id": uuid4(),
            "payload": {"sql": "SELECT 1"},
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        resolver.resolve(
            operation="db.query",
            bindings_subcontract=simple_bindings,
            envelope=envelope,
            context=context,
        )

        assert resolver.metrics.total_resolutions == 1

        # Reset
        resolver.reset_metrics()

        m = resolver.metrics
        assert m.total_resolutions == 0
        assert m.successful_resolutions == 0
        assert m.failed_resolutions == 0
        assert m.bindings_resolved_count == 0
        assert m.total_latency_ms == 0.0

    def test_metrics_to_dict_serializable(
        self,
        resolver: OperationBindingResolver,
        simple_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Metrics to_dict() produces JSON-serializable output."""
        import json

        envelope = {
            "correlation_id": uuid4(),
            "payload": {"sql": "SELECT 1"},
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        resolver.resolve(
            operation="db.query",
            bindings_subcontract=simple_bindings,
            envelope=envelope,
            context=context,
        )

        metrics_dict = resolver.metrics.to_dict()
        # Should not raise
        serialized = json.dumps(metrics_dict)
        assert isinstance(serialized, str)

        # Verify key fields are present
        assert "total_resolutions" in metrics_dict
        assert "latency_histogram" in metrics_dict
        assert "per_operation_resolutions" in metrics_dict
        assert "success_rate" in metrics_dict

    def test_avg_latency_computation(
        self,
        resolver: OperationBindingResolver,
        simple_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """avg_latency_ms is computed correctly from total_latency_ms / total_resolutions."""
        envelope = {
            "correlation_id": uuid4(),
            "payload": {"sql": "SELECT 1"},
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        for _ in range(5):
            resolver.resolve(
                operation="db.query",
                bindings_subcontract=simple_bindings,
                envelope=envelope,
                context=context,
            )

        m = resolver.metrics
        expected_avg = m.total_latency_ms / m.total_resolutions
        assert m.avg_latency_ms == pytest.approx(expected_avg)

    def test_bindings_resolved_count_tracks_individual_bindings(
        self,
        resolver: OperationBindingResolver,
        simple_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """bindings_resolved_count tracks total individual bindings, not resolutions."""
        envelope = {
            "correlation_id": uuid4(),
            "payload": {"sql": "SELECT 1"},
        }
        context = {"now_iso": "2026-01-01T00:00:00Z"}

        resolver.resolve(
            operation="db.query",
            bindings_subcontract=simple_bindings,
            envelope=envelope,
            context=context,
        )

        m = resolver.metrics
        # simple_bindings has 1 global binding + 2 operation bindings = 3 total
        assert m.bindings_resolved_count == 3, (
            f"Expected 3 bindings resolved (1 global + 2 operation), "
            f"got {m.bindings_resolved_count}"
        )
