# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelEffectivenessInvalidationEvent.

Tests the Pydantic model used for effectiveness data change notifications.

Related Tickets:
    - OMN-2303: Activate effectiveness consumer and populate measurement tables
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.services.observability.injection_effectiveness.models.model_invalidation_event import (
    ModelEffectivenessInvalidationEvent,
)


class TestModelEffectivenessInvalidationEvent:
    """Tests for ModelEffectivenessInvalidationEvent validation."""

    def test_valid_kafka_consumer_source(self) -> None:
        event = ModelEffectivenessInvalidationEvent(
            tables_affected=("injection_effectiveness",),
            rows_written=10,
            source="kafka_consumer",
        )
        assert event.source == "kafka_consumer"

    def test_valid_batch_compute_source(self) -> None:
        event = ModelEffectivenessInvalidationEvent(
            tables_affected=("latency_breakdowns", "pattern_hit_rates"),
            rows_written=5,
            source="batch_compute",
        )
        assert event.source == "batch_compute"

    def test_invalid_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelEffectivenessInvalidationEvent(
                tables_affected=("injection_effectiveness",),
                rows_written=1,
                source="unknown_source",  # type: ignore[arg-type]
            )

    def test_negative_rows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelEffectivenessInvalidationEvent(
                tables_affected=("injection_effectiveness",),
                rows_written=-1,
                source="kafka_consumer",
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelEffectivenessInvalidationEvent(
                tables_affected=("injection_effectiveness",),
                rows_written=1,
                source="kafka_consumer",
                extra_field="should fail",  # type: ignore[call-arg]
            )

    def test_custom_correlation_id(self) -> None:
        custom_id = uuid4()
        event = ModelEffectivenessInvalidationEvent(
            correlation_id=custom_id,
            tables_affected=("injection_effectiveness",),
            rows_written=1,
            source="batch_compute",
        )
        assert event.correlation_id == custom_id

    def test_default_event_type(self) -> None:
        event = ModelEffectivenessInvalidationEvent(
            tables_affected=("injection_effectiveness",),
            rows_written=1,
            source="kafka_consumer",
        )
        assert event.event_type == "effectiveness_data_changed"

    def test_empty_tables_rejected(self) -> None:
        """Empty tuple is semantically meaningless and must be rejected."""
        with pytest.raises(ValidationError):
            ModelEffectivenessInvalidationEvent(
                tables_affected=(),
                rows_written=0,
                source="batch_compute",
            )

    def test_unknown_table_name_rejected(self) -> None:
        """Validator rejects table names not in the known set."""
        with pytest.raises(ValidationError):
            ModelEffectivenessInvalidationEvent(
                tables_affected=("nonexistent_table",),
                rows_written=1,
                source="batch_compute",
            )

    def test_model_dump_json_mode(self) -> None:
        event = ModelEffectivenessInvalidationEvent(
            tables_affected=("injection_effectiveness", "latency_breakdowns"),
            rows_written=42,
            source="kafka_consumer",
        )
        data = event.model_dump(mode="json")
        assert isinstance(data["correlation_id"], str)
        assert isinstance(data["emitted_at"], str)
        assert data["tables_affected"] == [
            "injection_effectiveness",
            "latency_breakdowns",
        ]
