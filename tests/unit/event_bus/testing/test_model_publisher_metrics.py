# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelPublisherMetrics.

Tests the metrics model used by AdapterProtocolEventPublisherInmemory.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.event_bus.testing import ModelPublisherMetrics


class TestModelPublisherMetrics:
    """Tests for ModelPublisherMetrics model."""

    def test_default_values(self) -> None:
        """Verify all metrics default to zero/closed."""
        metrics = ModelPublisherMetrics()

        assert metrics.events_published == 0
        assert metrics.events_failed == 0
        assert metrics.events_sent_to_dlq == 0
        assert metrics.total_publish_time_ms == 0.0
        assert metrics.avg_publish_time_ms == 0.0
        assert metrics.circuit_breaker_opens == 0
        assert metrics.retries_attempted == 0
        assert metrics.circuit_breaker_status == "closed"
        assert metrics.current_failures == 0

    def test_to_dict_returns_all_fields(self) -> None:
        """Verify to_dict includes all metric fields."""
        metrics = ModelPublisherMetrics(
            events_published=10,
            events_failed=2,
            total_publish_time_ms=150.5,
            avg_publish_time_ms=15.05,
        )

        result = metrics.to_dict()

        assert result["events_published"] == 10
        assert result["events_failed"] == 2
        assert result["events_sent_to_dlq"] == 0
        assert result["total_publish_time_ms"] == 150.5
        assert result["avg_publish_time_ms"] == 15.05
        assert result["circuit_breaker_opens"] == 0
        assert result["retries_attempted"] == 0
        assert result["circuit_breaker_status"] == "closed"
        assert result["current_failures"] == 0

    def test_to_dict_keys_match_fields(self) -> None:
        """Verify to_dict keys match model field names."""
        metrics = ModelPublisherMetrics()
        result = metrics.to_dict()

        expected_keys = {
            "events_published",
            "events_failed",
            "events_sent_to_dlq",
            "total_publish_time_ms",
            "avg_publish_time_ms",
            "circuit_breaker_opens",
            "retries_attempted",
            "circuit_breaker_status",
            "current_failures",
        }

        assert set(result.keys()) == expected_keys

    def test_model_is_mutable(self) -> None:
        """Verify model can be mutated (frozen=False)."""
        metrics = ModelPublisherMetrics()

        # Should not raise - model is mutable
        metrics.events_published = 5
        metrics.events_failed = 1
        metrics.current_failures = 2

        assert metrics.events_published == 5
        assert metrics.events_failed == 1
        assert metrics.current_failures == 2

    def test_negative_values_rejected(self) -> None:
        """Verify negative values are rejected by ge=0 constraint."""
        with pytest.raises(ValidationError):
            ModelPublisherMetrics(events_published=-1)

        with pytest.raises(ValidationError):
            ModelPublisherMetrics(events_failed=-1)

        with pytest.raises(ValidationError):
            ModelPublisherMetrics(total_publish_time_ms=-0.1)

    def test_extra_fields_forbidden(self) -> None:
        """Verify extra fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError):
            ModelPublisherMetrics(unknown_field="value")
