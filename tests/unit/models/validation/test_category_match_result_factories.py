# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dedicated unit tests for ModelCategoryMatchResult factory methods.

This module provides comprehensive tests for the factory methods of
ModelCategoryMatchResult, ensuring each creates instances with the correct
field values and state. This is separate from the __bool__ behavior tests
in test_custom_bool_behavior.py.

Factory methods tested:
    - matched_with_category(category): Creates match with specific category
    - matched_without_category(): Creates match without category
    - not_matched(): Creates non-match result

Properties tested:
    - has_category: Whether category is not None
    - is_message_category: Whether category is EnumMessageCategory
    - is_projection: Whether category is EnumNodeOutputType.PROJECTION

.. versionadded:: 0.7.0
    Created as part of PR #92 review to add dedicated factory method tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
from omnibase_infra.models.validation.model_category_match_result import (
    ModelCategoryMatchResult,
)


class TestCategoryMatchResultMatchedWithCategoryFactory:
    """Tests for ModelCategoryMatchResult.matched_with_category() factory.

    This factory creates a successful match result with a specific category
    (EnumMessageCategory or EnumNodeOutputType).
    """

    def test_matched_with_event_category(self) -> None:
        """Verify matched_with_category creates correct instance for EVENT."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )

        assert result.matched is True
        assert result.category == EnumMessageCategory.EVENT
        assert result.has_category is True
        assert result.is_message_category is True
        assert result.is_projection is False

    def test_matched_with_command_category(self) -> None:
        """Verify matched_with_category creates correct instance for COMMAND."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.COMMAND
        )

        assert result.matched is True
        assert result.category == EnumMessageCategory.COMMAND
        assert result.has_category is True
        assert result.is_message_category is True
        assert result.is_projection is False

    def test_matched_with_intent_category(self) -> None:
        """Verify matched_with_category creates correct instance for INTENT."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.INTENT
        )

        assert result.matched is True
        assert result.category == EnumMessageCategory.INTENT
        assert result.has_category is True
        assert result.is_message_category is True
        assert result.is_projection is False

    def test_matched_with_projection_category(self) -> None:
        """Verify matched_with_category creates correct instance for PROJECTION.

        PROJECTION is a node output type, not a message category.
        """
        result = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.PROJECTION
        )

        assert result.matched is True
        assert result.category == EnumNodeOutputType.PROJECTION
        assert result.has_category is True
        assert result.is_message_category is False
        assert result.is_projection is True

    @pytest.mark.parametrize(
        ("category", "expected_is_message", "expected_is_projection"),
        [
            (EnumMessageCategory.EVENT, True, False),
            (EnumMessageCategory.COMMAND, True, False),
            (EnumMessageCategory.INTENT, True, False),
            (EnumNodeOutputType.PROJECTION, False, True),
        ],
        ids=["event", "command", "intent", "projection"],
    )
    def test_matched_with_category_parametrized(
        self,
        category: EnumMessageCategory | EnumNodeOutputType,
        expected_is_message: bool,
        expected_is_projection: bool,
    ) -> None:
        """Parametrized test covering all category types."""
        result = ModelCategoryMatchResult.matched_with_category(category)

        assert result.matched is True
        assert result.category == category
        assert result.has_category is True
        assert result.is_message_category is expected_is_message
        assert result.is_projection is expected_is_projection


class TestCategoryMatchResultMatchedWithoutCategoryFactory:
    """Tests for ModelCategoryMatchResult.matched_without_category() factory.

    This factory creates a successful match result without a specific category.
    Used when a pattern indicates a message type but doesn't specify which category.
    """

    def test_matched_without_category_creates_correct_instance(self) -> None:
        """Verify matched_without_category sets matched=True, category=None."""
        result = ModelCategoryMatchResult.matched_without_category()

        assert result.matched is True
        assert result.category is None

    def test_matched_without_category_has_category_is_false(self) -> None:
        """Verify has_category property returns False when category is None."""
        result = ModelCategoryMatchResult.matched_without_category()

        assert result.has_category is False

    def test_matched_without_category_is_message_category_is_false(self) -> None:
        """Verify is_message_category returns False when category is None."""
        result = ModelCategoryMatchResult.matched_without_category()

        assert result.is_message_category is False

    def test_matched_without_category_is_projection_is_false(self) -> None:
        """Verify is_projection returns False when category is None."""
        result = ModelCategoryMatchResult.matched_without_category()

        assert result.is_projection is False

    def test_matched_without_category_bool_is_true(self) -> None:
        """Verify __bool__ returns True even without category (match occurred)."""
        result = ModelCategoryMatchResult.matched_without_category()

        assert bool(result) is True


class TestCategoryMatchResultNotMatchedFactory:
    """Tests for ModelCategoryMatchResult.not_matched() factory.

    This factory creates a result indicating no match was found.
    """

    def test_not_matched_creates_correct_instance(self) -> None:
        """Verify not_matched sets matched=False, category=None."""
        result = ModelCategoryMatchResult.not_matched()

        assert result.matched is False
        assert result.category is None

    def test_not_matched_has_category_is_false(self) -> None:
        """Verify has_category property returns False."""
        result = ModelCategoryMatchResult.not_matched()

        assert result.has_category is False

    def test_not_matched_is_message_category_is_false(self) -> None:
        """Verify is_message_category returns False."""
        result = ModelCategoryMatchResult.not_matched()

        assert result.is_message_category is False

    def test_not_matched_is_projection_is_false(self) -> None:
        """Verify is_projection returns False."""
        result = ModelCategoryMatchResult.not_matched()

        assert result.is_projection is False

    def test_not_matched_bool_is_false(self) -> None:
        """Verify __bool__ returns False for non-match."""
        result = ModelCategoryMatchResult.not_matched()

        assert bool(result) is False


class TestCategoryMatchResultFactoryReturnTypes:
    """Tests verifying factory methods return correct type."""

    def test_matched_with_category_returns_model_instance(self) -> None:
        """Verify matched_with_category returns ModelCategoryMatchResult."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )
        assert isinstance(result, ModelCategoryMatchResult)

    def test_matched_without_category_returns_model_instance(self) -> None:
        """Verify matched_without_category returns ModelCategoryMatchResult."""
        result = ModelCategoryMatchResult.matched_without_category()
        assert isinstance(result, ModelCategoryMatchResult)

    def test_not_matched_returns_model_instance(self) -> None:
        """Verify not_matched returns ModelCategoryMatchResult."""
        result = ModelCategoryMatchResult.not_matched()
        assert isinstance(result, ModelCategoryMatchResult)


class TestCategoryMatchResultModelConfiguration:
    """Tests for model configuration (frozen, extra forbid)."""

    def test_model_is_frozen(self) -> None:
        """Verify model instances are immutable (frozen=True)."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )

        with pytest.raises(ValidationError):
            result.matched = False  # type: ignore[misc]

    def test_model_forbids_extra_fields(self) -> None:
        """Verify model rejects extra fields (extra='forbid')."""
        with pytest.raises(ValidationError):
            ModelCategoryMatchResult(
                matched=True,
                category=EnumMessageCategory.EVENT,
                extra_field="not allowed",  # type: ignore[call-arg]
            )


class TestCategoryMatchResultDirectConstruction:
    """Tests for direct model construction (not via factory)."""

    def test_direct_construction_matched_with_category(self) -> None:
        """Verify direct construction works with matched=True and category."""
        result = ModelCategoryMatchResult(
            matched=True,
            category=EnumMessageCategory.COMMAND,
        )

        assert result.matched is True
        assert result.category == EnumMessageCategory.COMMAND

    def test_direct_construction_matched_without_category(self) -> None:
        """Verify direct construction works with matched=True and no category."""
        result = ModelCategoryMatchResult(matched=True, category=None)

        assert result.matched is True
        assert result.category is None

    def test_direct_construction_not_matched(self) -> None:
        """Verify direct construction works with matched=False."""
        result = ModelCategoryMatchResult(matched=False, category=None)

        assert result.matched is False
        assert result.category is None

    def test_direct_construction_category_defaults_to_none(self) -> None:
        """Verify category field defaults to None when not provided."""
        result = ModelCategoryMatchResult(matched=True)

        assert result.category is None


class TestCategoryMatchResultPropertyConsistency:
    """Tests verifying properties are consistent across factory methods."""

    def test_has_category_consistency(self) -> None:
        """Verify has_category is consistent with category field."""
        # With category
        with_category = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )
        assert with_category.has_category == (with_category.category is not None)

        # Without category (matched)
        without_category = ModelCategoryMatchResult.matched_without_category()
        assert without_category.has_category == (without_category.category is not None)

        # Not matched
        not_matched = ModelCategoryMatchResult.not_matched()
        assert not_matched.has_category == (not_matched.category is not None)

    def test_is_message_category_only_true_for_enum_message_category(self) -> None:
        """Verify is_message_category only True for EnumMessageCategory values."""
        # All EnumMessageCategory values should return True
        for category in EnumMessageCategory:
            result = ModelCategoryMatchResult.matched_with_category(category)
            assert result.is_message_category is True

        # PROJECTION should return False
        projection = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.PROJECTION
        )
        assert projection.is_message_category is False

        # None category should return False
        no_category = ModelCategoryMatchResult.matched_without_category()
        assert no_category.is_message_category is False

    def test_is_projection_only_true_for_projection(self) -> None:
        """Verify is_projection only True for PROJECTION value."""
        # PROJECTION should return True
        projection = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.PROJECTION
        )
        assert projection.is_projection is True

        # EnumMessageCategory values should return False
        for category in EnumMessageCategory:
            result = ModelCategoryMatchResult.matched_with_category(category)
            assert result.is_projection is False

        # None category should return False
        no_category = ModelCategoryMatchResult.matched_without_category()
        assert no_category.is_projection is False


class TestCategoryMatchResultEdgeCases:
    """Edge case tests for factory methods."""

    def test_factory_methods_create_distinct_instances(self) -> None:
        """Verify each factory call creates a new instance."""
        result1 = ModelCategoryMatchResult.not_matched()
        result2 = ModelCategoryMatchResult.not_matched()

        # Should be equal but not the same object
        assert result1 == result2
        assert result1 is not result2

    def test_same_category_creates_equal_instances(self) -> None:
        """Verify same category produces equal instances."""
        result1 = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )
        result2 = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )

        assert result1 == result2

    def test_different_categories_create_unequal_instances(self) -> None:
        """Verify different categories produce unequal instances."""
        event = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )
        command = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.COMMAND
        )

        assert event != command

    def test_matched_vs_not_matched_are_unequal(self) -> None:
        """Verify matched and not_matched results are unequal."""
        matched = ModelCategoryMatchResult.matched_without_category()
        not_matched = ModelCategoryMatchResult.not_matched()

        # Both have category=None, but differ in matched
        assert matched.category == not_matched.category
        assert matched != not_matched

    def test_enum_node_output_type_event_command_intent(self) -> None:
        """Verify EnumNodeOutputType EVENT/COMMAND/INTENT work correctly.

        EnumNodeOutputType includes EVENT, COMMAND, INTENT, PROJECTION.
        The first three should behave like EnumMessageCategory for is_message_category,
        but technically they are EnumNodeOutputType, so is_message_category is False.
        """
        # These are EnumNodeOutputType, not EnumMessageCategory
        event_output = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.EVENT
        )
        command_output = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.COMMAND
        )
        intent_output = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.INTENT
        )

        # All should have matched=True and has_category=True
        assert event_output.matched is True
        assert event_output.has_category is True
        assert command_output.matched is True
        assert command_output.has_category is True
        assert intent_output.matched is True
        assert intent_output.has_category is True

        # But is_message_category should be False (they're EnumNodeOutputType)
        assert event_output.is_message_category is False
        assert command_output.is_message_category is False
        assert intent_output.is_message_category is False

        # And is_projection should be False (not PROJECTION)
        assert event_output.is_projection is False
        assert command_output.is_projection is False
        assert intent_output.is_projection is False
