# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelPricingEntry.

Related Tickets:
    - OMN-2239: E1-T3 Model pricing table and cost estimation
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.pricing.model_pricing_entry import ModelPricingEntry


@pytest.mark.unit
class TestModelPricingEntry:
    """Tests for ModelPricingEntry validation and immutability."""

    def test_valid_cloud_model_entry(self) -> None:
        """A cloud model with non-zero costs should validate successfully."""
        entry = ModelPricingEntry(
            input_cost_per_1k=0.015,
            output_cost_per_1k=0.075,
            effective_date="2026-02-01",
        )
        assert entry.input_cost_per_1k == 0.015
        assert entry.output_cost_per_1k == 0.075
        assert entry.effective_date == "2026-02-01"
        assert entry.note == ""

    def test_valid_local_model_entry(self) -> None:
        """A local model with zero costs and a note should validate."""
        entry = ModelPricingEntry(
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            effective_date="2026-02-01",
            note="Local model - zero API cost",
        )
        assert entry.input_cost_per_1k == 0.0
        assert entry.output_cost_per_1k == 0.0
        assert entry.note == "Local model - zero API cost"

    def test_negative_input_cost_rejected(self) -> None:
        """Negative input_cost_per_1k should fail validation."""
        with pytest.raises(ValidationError, match="input_cost_per_1k"):
            ModelPricingEntry(
                input_cost_per_1k=-0.001,
                output_cost_per_1k=0.0,
                effective_date="2026-02-01",
            )

    def test_negative_output_cost_rejected(self) -> None:
        """Negative output_cost_per_1k should fail validation."""
        with pytest.raises(ValidationError, match="output_cost_per_1k"):
            ModelPricingEntry(
                input_cost_per_1k=0.0,
                output_cost_per_1k=-0.001,
                effective_date="2026-02-01",
            )

    def test_invalid_date_format_rejected(self) -> None:
        """Non-ISO-8601 date format should fail validation."""
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            ModelPricingEntry(
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                effective_date="02/01/2026",
            )

    def test_date_too_short_rejected(self) -> None:
        """Date shorter than 10 chars should fail validation."""
        with pytest.raises(ValidationError):
            ModelPricingEntry(
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                effective_date="2026-2-1",
            )

    def test_extra_fields_rejected(self) -> None:
        """Extra fields should be rejected (extra='forbid')."""
        with pytest.raises(ValidationError, match="extra"):
            ModelPricingEntry(
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                effective_date="2026-02-01",
                unknown_field="oops",  # type: ignore[call-arg]
            )

    def test_frozen_immutability(self) -> None:
        """Entry should be immutable (frozen=True)."""
        entry = ModelPricingEntry(
            input_cost_per_1k=0.015,
            output_cost_per_1k=0.075,
            effective_date="2026-02-01",
        )
        with pytest.raises(ValidationError):
            entry.input_cost_per_1k = 0.02  # type: ignore[misc]
