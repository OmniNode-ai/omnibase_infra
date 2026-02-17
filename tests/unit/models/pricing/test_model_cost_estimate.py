# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelCostEstimate.

Related Tickets:
    - OMN-2239: E1-T3 Model pricing table and cost estimation
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.pricing.model_cost_estimate import ModelCostEstimate


@pytest.mark.unit
class TestModelCostEstimate:
    """Tests for ModelCostEstimate validation and semantics."""

    def test_known_model_with_cost(self) -> None:
        """A known model should have a non-None estimated_cost_usd."""
        estimate = ModelCostEstimate(
            model_id="claude-opus-4-6",
            prompt_tokens=1000,
            completion_tokens=500,
            estimated_cost_usd=0.0525,
        )
        assert estimate.estimated_cost_usd == 0.0525
        assert estimate.model_id == "claude-opus-4-6"

    def test_unknown_model_returns_none(self) -> None:
        """An unknown model should have estimated_cost_usd=None."""
        estimate = ModelCostEstimate(
            model_id="unknown-model",
            prompt_tokens=1000,
            completion_tokens=500,
            estimated_cost_usd=None,
        )
        assert estimate.estimated_cost_usd is None

    def test_local_model_returns_zero(self) -> None:
        """A local model should have estimated_cost_usd=0.0 (not None)."""
        estimate = ModelCostEstimate(
            model_id="qwen2.5-coder-14b",
            prompt_tokens=1000,
            completion_tokens=500,
            estimated_cost_usd=0.0,
        )
        assert estimate.estimated_cost_usd == 0.0
        assert estimate.estimated_cost_usd is not None

    def test_default_cost_is_none(self) -> None:
        """Default estimated_cost_usd should be None."""
        estimate = ModelCostEstimate(
            model_id="test-model",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert estimate.estimated_cost_usd is None

    def test_negative_tokens_rejected(self) -> None:
        """Negative token counts should fail validation."""
        with pytest.raises(ValidationError, match="prompt_tokens"):
            ModelCostEstimate(
                model_id="test",
                prompt_tokens=-1,
                completion_tokens=0,
            )

    def test_empty_model_id_rejected(self) -> None:
        """Empty model_id should fail validation."""
        with pytest.raises(ValidationError, match="model_id"):
            ModelCostEstimate(
                model_id="",
                prompt_tokens=0,
                completion_tokens=0,
            )

    def test_frozen_immutability(self) -> None:
        """Estimate should be immutable (frozen=True)."""
        estimate = ModelCostEstimate(
            model_id="test",
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost_usd=0.01,
        )
        with pytest.raises(ValidationError):
            estimate.estimated_cost_usd = 0.02  # type: ignore[misc]
