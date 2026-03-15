# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for NodeDeltaMetricsEffect payload models.

Tests model validation, serialization, and immutability for
ModelPayloadUpsertMetrics.

Related Tickets:
    - OMN-3142: NodeDeltaMetricsEffect implementation
"""

from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_delta_metrics_effect.models.model_payload_upsert_metrics import (
    ModelPayloadUpsertMetrics,
)


@pytest.mark.unit
class TestModelPayloadUpsertMetrics:
    """Tests for ModelPayloadUpsertMetrics."""

    def test_minimal_valid_payload(self) -> None:
        """Minimal required fields produce a valid payload."""
        payload = ModelPayloadUpsertMetrics(
            correlation_id=uuid4(),
            coding_model="claude-opus-4-20250514",
            subsystem="omnibase_infra",
            outcome="merged",
            gate_decision="PASS",
            period_start=date(2026, 3, 1),
            period_end=date(2026, 3, 7),
        )
        assert payload.intent_type == "delta_metrics.upsert_metrics"
        assert payload.is_fix_pr is False
        assert payload.gate_violation_count == 0

    def test_full_payload(self) -> None:
        """All fields populated produce a valid payload."""
        payload = ModelPayloadUpsertMetrics(
            correlation_id=uuid4(),
            coding_model="claude-opus-4-20250514",
            subsystem="omnibase_infra",
            outcome="reverted",
            gate_decision="QUARANTINE",
            is_fix_pr=True,
            gate_violation_count=3,
            period_start=date(2026, 3, 1),
            period_end=date(2026, 3, 7),
        )
        assert payload.outcome == "reverted"
        assert payload.gate_decision == "QUARANTINE"
        assert payload.is_fix_pr is True
        assert payload.gate_violation_count == 3

    def test_frozen_model(self) -> None:
        """Model is frozen (immutable)."""
        payload = ModelPayloadUpsertMetrics(
            correlation_id=uuid4(),
            coding_model="model",
            subsystem="sub",
            outcome="merged",
            gate_decision="PASS",
            period_start=date(2026, 1, 1),
            period_end=date(2026, 1, 7),
        )
        with pytest.raises(ValidationError):
            payload.coding_model = "changed"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelPayloadUpsertMetrics(
                correlation_id=uuid4(),
                coding_model="model",
                subsystem="sub",
                outcome="merged",
                gate_decision="PASS",
                period_start=date(2026, 1, 1),
                period_end=date(2026, 1, 7),
                extra_field="boom",  # type: ignore[call-arg]
            )

    def test_invalid_outcome(self) -> None:
        """Invalid outcome value is rejected."""
        with pytest.raises(ValidationError):
            ModelPayloadUpsertMetrics(
                correlation_id=uuid4(),
                coding_model="model",
                subsystem="sub",
                outcome="abandoned",  # type: ignore[arg-type]
                gate_decision="PASS",
                period_start=date(2026, 1, 1),
                period_end=date(2026, 1, 7),
            )

    def test_invalid_gate_decision(self) -> None:
        """Invalid gate_decision value is rejected."""
        with pytest.raises(ValidationError):
            ModelPayloadUpsertMetrics(
                correlation_id=uuid4(),
                coding_model="model",
                subsystem="sub",
                outcome="merged",
                gate_decision="INVALID",  # type: ignore[arg-type]
                period_start=date(2026, 1, 1),
                period_end=date(2026, 1, 7),
            )
