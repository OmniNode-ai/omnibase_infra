# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelDependencySpec.

Tests validation behavior for dependency specifications, including
selection strategy validation that rejects unimplemented strategies.

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery for capability-based discovery
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.discovery.model_dependency_spec import ModelDependencySpec


class TestModelDependencySpecSelectionStrategy:
    """Test selection strategy validation in ModelDependencySpec."""

    def test_least_loaded_strategy_rejected(self) -> None:
        """Test that least_loaded strategy raises ValidationError.

        The LEAST_LOADED strategy is reserved for future use and should
        be rejected at model creation time with a clear error message.
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelDependencySpec(
                name="test_dep",
                type="node",
                capability="test.capability",
                selection_strategy="least_loaded",
            )

        # Verify the error message is helpful
        error_str = str(exc_info.value)
        assert "LEAST_LOADED selection strategy is not yet implemented" in error_str
        assert "first" in error_str
        assert "random" in error_str
        assert "round_robin" in error_str

    @pytest.mark.parametrize(
        "strategy",
        ["first", "random", "round_robin"],
    )
    def test_implemented_strategies_accepted(self, strategy: str) -> None:
        """Test that implemented strategies are accepted.

        Args:
            strategy: The selection strategy to test.
        """
        spec = ModelDependencySpec(
            name="test_dep",
            type="node",
            capability="test.capability",
            selection_strategy=strategy,
        )
        assert spec.selection_strategy == strategy

    def test_default_strategy_is_first(self) -> None:
        """Test that the default selection strategy is 'first'."""
        spec = ModelDependencySpec(
            name="test_dep",
            type="node",
            capability="test.capability",
        )
        assert spec.selection_strategy == "first"

    def test_invalid_strategy_rejected_by_literal(self) -> None:
        """Test that completely invalid strategies are rejected by Literal type."""
        with pytest.raises(ValidationError) as exc_info:
            ModelDependencySpec(
                name="test_dep",
                type="node",
                capability="test.capability",
                selection_strategy="invalid_strategy",  # type: ignore[arg-type]
            )

        error_str = str(exc_info.value)
        assert "selection_strategy" in error_str


class TestModelDependencySpecFilterValidation:
    """Test filter validation in ModelDependencySpec."""

    def test_no_filter_raises_error(self) -> None:
        """Test that a spec without any filter raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelDependencySpec(
                name="test_dep",
                type="node",
            )

        error_str = str(exc_info.value)
        assert "must have at least one discovery filter" in error_str

    def test_capability_filter_accepted(self) -> None:
        """Test that capability filter alone is sufficient."""
        spec = ModelDependencySpec(
            name="test_dep",
            type="node",
            capability="test.capability",
        )
        assert spec.has_capability_filter()
        assert spec.has_any_filter()

    def test_intent_types_filter_accepted(self) -> None:
        """Test that intent_types filter alone is sufficient."""
        spec = ModelDependencySpec(
            name="test_dep",
            type="node",
            intent_types=["test.intent"],
        )
        assert spec.has_intent_filter()
        assert spec.has_any_filter()

    def test_protocol_filter_accepted(self) -> None:
        """Test that protocol filter alone is sufficient."""
        spec = ModelDependencySpec(
            name="test_dep",
            type="protocol",
            protocol="ProtocolTest",
        )
        assert spec.has_protocol_filter()
        assert spec.has_any_filter()

    def test_multiple_filters_accepted(self) -> None:
        """Test that multiple filters can be combined."""
        spec = ModelDependencySpec(
            name="test_dep",
            type="node",
            capability="test.capability",
            intent_types=["test.intent"],
            protocol="ProtocolTest",
        )
        assert spec.has_capability_filter()
        assert spec.has_intent_filter()
        assert spec.has_protocol_filter()
        assert spec.has_any_filter()


__all__: list[str] = [
    "TestModelDependencySpecSelectionStrategy",
    "TestModelDependencySpecFilterValidation",
]
