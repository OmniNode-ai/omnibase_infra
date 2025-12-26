# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for custom __bool__ behavior in runtime optional models.

This test module verifies the non-standard __bool__ behavior of Pydantic models
that override the default behavior. Standard Pydantic models always return True
when evaluated in a boolean context (bool(model) == True for any valid instance).
However, these optional wrapper models override __bool__ to enable idiomatic
presence checks.

**IMPORTANT**: This behavior differs from typical Pydantic where bool(model) is always True.

Models covered:
    - ModelOptionalString: True if value is present (not None)
    - ModelOptionalUUID: True if value is present (not None)
    - ModelOptionalCorrelationId: True if value is present (not None)
    - ModelPolicyTypeFilter: True if a filter value is set

See Also:
    CLAUDE.md section "Custom `__bool__` for Result Models" for documentation standards.

.. versionadded:: 0.7.0
    Created as part of PR #92 review to add missing test coverage for __bool__.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.runtime.models.model_optional_correlation_id import (
    ModelOptionalCorrelationId,
)
from omnibase_infra.runtime.models.model_optional_string import ModelOptionalString
from omnibase_infra.runtime.models.model_optional_uuid import ModelOptionalUUID
from omnibase_infra.runtime.models.model_policy_type_filter import ModelPolicyTypeFilter

# =============================================================================
# Tests for ModelOptionalString.__bool__
# =============================================================================


@pytest.mark.unit
class TestModelOptionalStringBool:
    """Tests for ModelOptionalString.__bool__ non-standard behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when value is present.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with value=None evaluates to False in boolean context!

    The design enables idiomatic presence checks:
        if opt_string:
            # Value is present
            use_value(opt_string.value)
        else:
            # No value
            use_default()
    """

    def test_bool_true_when_has_value(self) -> None:
        """Result evaluates to True when value is present."""
        opt = ModelOptionalString(value="hello")

        assert bool(opt) is True
        assert opt.has_value() is True
        assert opt  # Direct conditional check

    def test_bool_false_when_no_value(self) -> None:
        """Result evaluates to False when value is None.

        WARNING: This is non-standard Pydantic behavior!
        A valid model instance returns False because there's no value.
        """
        opt = ModelOptionalString(value=None)

        assert bool(opt) is False
        assert opt.has_value() is False
        assert not opt  # Direct conditional check

    def test_bool_false_for_default_construction(self) -> None:
        """Default construction (no value) evaluates to False."""
        opt = ModelOptionalString()

        assert bool(opt) is False
        assert opt.value is None

    def test_bool_true_for_empty_string(self) -> None:
        """Empty string is still a value, so evaluates to True.

        This is an important distinction: None means "no value",
        while "" means "a value that happens to be empty".
        """
        opt = ModelOptionalString(value="")

        assert bool(opt) is True
        assert opt.has_value() is True
        assert opt.value == ""

    def test_has_value_matches_bool(self) -> None:
        """has_value() should match __bool__ return value."""
        # Falsy case: no value
        empty = ModelOptionalString()
        assert empty.has_value() == bool(empty)
        assert empty.has_value() is False

        # Truthy case: has value
        with_value = ModelOptionalString(value="test")
        assert with_value.has_value() == bool(with_value)
        assert with_value.has_value() is True

    def test_conditional_pattern_usage(self) -> None:
        """Demonstrate the idiomatic conditional usage pattern."""
        with_value = ModelOptionalString(value="present")
        without_value = ModelOptionalString()

        # Pattern 1: Value present
        result = None
        if with_value:
            result = with_value.value
        assert result == "present"

        # Pattern 2: Value absent
        used_default = False
        if not without_value:
            used_default = True
        assert used_default is True

    def test_get_or_default_behavior(self) -> None:
        """Verify get_or_default works with boolean context."""
        opt = ModelOptionalString()

        if not opt:
            value = opt.get_or_default("fallback")
        else:
            value = opt.value

        assert value == "fallback"

    def test_model_is_frozen(self) -> None:
        """Verify model is immutable (frozen=True)."""
        opt = ModelOptionalString(value="test")

        with pytest.raises(ValidationError):
            opt.value = "changed"  # type: ignore[misc]


# =============================================================================
# Tests for ModelOptionalUUID.__bool__
# =============================================================================


@pytest.mark.unit
class TestModelOptionalUUIDBool:
    """Tests for ModelOptionalUUID.__bool__ non-standard behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when value is present.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with value=None evaluates to False in boolean context!

    The design enables idiomatic presence checks:
        if opt_uuid:
            # UUID is present
            use_uuid(opt_uuid.value)
        else:
            # No UUID
            generate_new()
    """

    def test_bool_true_when_has_value(self) -> None:
        """Result evaluates to True when UUID is present."""
        test_uuid = uuid4()
        opt = ModelOptionalUUID(value=test_uuid)

        assert bool(opt) is True
        assert opt.has_value() is True
        assert opt  # Direct conditional check

    def test_bool_false_when_no_value(self) -> None:
        """Result evaluates to False when value is None.

        WARNING: This is non-standard Pydantic behavior!
        A valid model instance returns False because there's no UUID.
        """
        opt = ModelOptionalUUID(value=None)

        assert bool(opt) is False
        assert opt.has_value() is False
        assert not opt  # Direct conditional check

    def test_bool_false_for_default_construction(self) -> None:
        """Default construction (no value) evaluates to False."""
        opt = ModelOptionalUUID()

        assert bool(opt) is False
        assert opt.value is None

    def test_has_value_matches_bool(self) -> None:
        """has_value() should match __bool__ return value."""
        # Falsy case: no value
        empty = ModelOptionalUUID()
        assert empty.has_value() == bool(empty)
        assert empty.has_value() is False

        # Truthy case: has value
        with_value = ModelOptionalUUID(value=uuid4())
        assert with_value.has_value() == bool(with_value)
        assert with_value.has_value() is True

    def test_conditional_pattern_usage(self) -> None:
        """Demonstrate the idiomatic conditional usage pattern."""
        test_uuid = uuid4()
        with_value = ModelOptionalUUID(value=test_uuid)
        without_value = ModelOptionalUUID()

        # Pattern 1: UUID present
        result = None
        if with_value:
            result = with_value.value
        assert result == test_uuid

        # Pattern 2: UUID absent
        generated = False
        if not without_value:
            generated = True
        assert generated is True

    def test_get_or_default_behavior(self) -> None:
        """Verify get_or_default works with boolean context."""
        opt = ModelOptionalUUID()
        default_uuid = uuid4()

        if not opt:
            value = opt.get_or_default(default_uuid)
        else:
            value = opt.value

        assert value == default_uuid

    def test_model_is_frozen(self) -> None:
        """Verify model is immutable (frozen=True)."""
        opt = ModelOptionalUUID(value=uuid4())

        with pytest.raises(ValidationError):
            opt.value = uuid4()  # type: ignore[misc]


# =============================================================================
# Tests for ModelOptionalCorrelationId.__bool__
# =============================================================================


@pytest.mark.unit
class TestModelOptionalCorrelationIdBool:
    """Tests for ModelOptionalCorrelationId.__bool__ non-standard behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when correlation ID is present.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with value=None evaluates to False in boolean context!

    The design enables idiomatic presence checks:
        if corr_id:
            # Propagate existing ID
            context.set_correlation_id(corr_id.value)
        else:
            # Generate new one
            corr_id = corr_id.get_or_generate()
    """

    def test_bool_true_when_has_value(self) -> None:
        """Result evaluates to True when correlation ID is present."""
        corr_id = ModelOptionalCorrelationId.generate()

        assert bool(corr_id) is True
        assert corr_id.has_value() is True
        assert corr_id  # Direct conditional check

    def test_bool_false_when_no_value(self) -> None:
        """Result evaluates to False when value is None.

        WARNING: This is non-standard Pydantic behavior!
        A valid model instance returns False because there's no correlation ID.
        """
        corr_id = ModelOptionalCorrelationId(value=None)

        assert bool(corr_id) is False
        assert corr_id.has_value() is False
        assert not corr_id  # Direct conditional check

    def test_bool_false_for_default_construction(self) -> None:
        """Default construction (no value) evaluates to False."""
        corr_id = ModelOptionalCorrelationId()

        assert bool(corr_id) is False
        assert corr_id.value is None

    def test_bool_false_for_none_factory(self) -> None:
        """ModelOptionalCorrelationId.none() evaluates to False."""
        corr_id = ModelOptionalCorrelationId.none()

        assert bool(corr_id) is False
        assert corr_id.value is None

    def test_bool_true_for_generate_factory(self) -> None:
        """ModelOptionalCorrelationId.generate() evaluates to True."""
        corr_id = ModelOptionalCorrelationId.generate()

        assert bool(corr_id) is True
        assert corr_id.value is not None
        assert isinstance(corr_id.value, UUID)

    def test_bool_true_for_from_uuid_factory(self) -> None:
        """ModelOptionalCorrelationId.from_uuid() evaluates to True."""
        test_uuid = uuid4()
        corr_id = ModelOptionalCorrelationId.from_uuid(test_uuid)

        assert bool(corr_id) is True
        assert corr_id.value == test_uuid

    def test_has_value_matches_bool(self) -> None:
        """has_value() should match __bool__ return value."""
        # Falsy case: no value
        empty = ModelOptionalCorrelationId.none()
        assert empty.has_value() == bool(empty)
        assert empty.has_value() is False

        # Truthy case: has value
        with_value = ModelOptionalCorrelationId.generate()
        assert with_value.has_value() == bool(with_value)
        assert with_value.has_value() is True

    def test_conditional_pattern_with_get_or_generate(self) -> None:
        """Demonstrate the idiomatic get_or_generate pattern."""
        empty = ModelOptionalCorrelationId.none()

        # If no correlation ID, generate one
        if not empty:
            filled = empty.get_or_generate()
        else:
            filled = empty

        assert bool(filled) is True
        assert filled.has_value() is True

    def test_get_or_generate_preserves_existing_value(self) -> None:
        """get_or_generate returns same instance if value exists."""
        original = ModelOptionalCorrelationId.generate()
        original_uuid = original.value

        result = original.get_or_generate()

        assert result.value == original_uuid

    def test_get_or_generate_generates_when_empty(self) -> None:
        """get_or_generate creates new UUID when empty."""
        empty = ModelOptionalCorrelationId.none()

        result = empty.get_or_generate()

        assert bool(result) is True
        assert result.value is not None

    def test_model_is_frozen(self) -> None:
        """Verify model is immutable (frozen=True)."""
        corr_id = ModelOptionalCorrelationId.generate()

        with pytest.raises(ValidationError):
            corr_id.value = uuid4()  # type: ignore[misc]


# =============================================================================
# Tests for ModelPolicyTypeFilter.__bool__
# =============================================================================


@pytest.mark.unit
class TestModelPolicyTypeFilterBool:
    """Tests for ModelPolicyTypeFilter.__bool__ non-standard behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when a filter value is set.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with no filter value evaluates to False in boolean context!

    The design enables idiomatic filter checks:
        if policy_filter:
            # Filter is active - apply it
            filtered = [p for p in policies if policy_filter.matches(p.type)]
        else:
            # No filter - return all
            filtered = policies
    """

    def test_bool_true_when_has_string_value(self) -> None:
        """Result evaluates to True when string filter is set."""
        filter_obj = ModelPolicyTypeFilter(string_value="orchestrator")

        assert bool(filter_obj) is True
        assert filter_obj.has_value() is True
        assert filter_obj  # Direct conditional check

    def test_bool_true_when_has_enum_value(self) -> None:
        """Result evaluates to True when enum filter is set."""
        filter_obj = ModelPolicyTypeFilter(enum_value=EnumPolicyType.REDUCER)

        assert bool(filter_obj) is True
        assert filter_obj.has_value() is True
        assert filter_obj  # Direct conditional check

    def test_bool_false_when_no_value(self) -> None:
        """Result evaluates to False when no filter value is set.

        WARNING: This is non-standard Pydantic behavior!
        A valid model instance returns False because no filter is active.
        """
        filter_obj = ModelPolicyTypeFilter()

        assert bool(filter_obj) is False
        assert filter_obj.has_value() is False
        assert not filter_obj  # Direct conditional check

    def test_bool_false_for_none_factory(self) -> None:
        """ModelPolicyTypeFilter.none() evaluates to False."""
        filter_obj = ModelPolicyTypeFilter.none()

        assert bool(filter_obj) is False
        assert filter_obj.has_value() is False

    def test_bool_true_for_from_string_factory(self) -> None:
        """ModelPolicyTypeFilter.from_string() evaluates to True."""
        filter_obj = ModelPolicyTypeFilter.from_string("effect")

        assert bool(filter_obj) is True
        assert filter_obj.string_value == "effect"

    def test_bool_true_for_from_enum_factory(self) -> None:
        """ModelPolicyTypeFilter.from_enum() evaluates to True."""
        filter_obj = ModelPolicyTypeFilter.from_enum(EnumPolicyType.ORCHESTRATOR)

        assert bool(filter_obj) is True
        assert filter_obj.enum_value == EnumPolicyType.ORCHESTRATOR

    def test_has_value_matches_bool(self) -> None:
        """has_value() should match __bool__ return value."""
        # Falsy case: no filter
        empty = ModelPolicyTypeFilter.none()
        assert empty.has_value() == bool(empty)
        assert empty.has_value() is False

        # Truthy case: has string filter
        with_string = ModelPolicyTypeFilter.from_string("compute")
        assert with_string.has_value() == bool(with_string)
        assert with_string.has_value() is True

        # Truthy case: has enum filter
        with_enum = ModelPolicyTypeFilter.from_enum(EnumPolicyType.ORCHESTRATOR)
        assert with_enum.has_value() == bool(with_enum)
        assert with_enum.has_value() is True

    def test_conditional_filter_pattern(self) -> None:
        """Demonstrate the idiomatic filter pattern."""
        active_filter = ModelPolicyTypeFilter.from_string("reducer")
        no_filter = ModelPolicyTypeFilter.none()

        # Pattern 1: Active filter
        filter_applied = False
        if active_filter:
            filter_applied = True
        assert filter_applied is True

        # Pattern 2: No filter - return all
        return_all = False
        if not no_filter:
            return_all = True
        assert return_all is True

    def test_matches_with_no_filter_returns_true(self) -> None:
        """Empty filter matches everything."""
        no_filter = ModelPolicyTypeFilter.none()

        assert no_filter.matches("any_type") is True
        assert no_filter.matches("orchestrator") is True
        assert no_filter.matches("reducer") is True

    def test_matches_with_filter_value(self) -> None:
        """Active filter only matches specific types."""
        filter_obj = ModelPolicyTypeFilter.from_string("orchestrator")

        assert filter_obj.matches("orchestrator") is True
        assert filter_obj.matches("reducer") is False
        assert filter_obj.matches("effect") is False

    def test_normalize_returns_enum_value(self) -> None:
        """normalize() returns enum value when enum is set."""
        filter_obj = ModelPolicyTypeFilter.from_enum(EnumPolicyType.REDUCER)

        assert filter_obj.normalize() == "reducer"

    def test_normalize_returns_string_value(self) -> None:
        """normalize() returns string value when string is set."""
        filter_obj = ModelPolicyTypeFilter.from_string("custom_type")

        assert filter_obj.normalize() == "custom_type"

    def test_normalize_returns_none_when_empty(self) -> None:
        """normalize() returns None when no value is set."""
        filter_obj = ModelPolicyTypeFilter.none()

        assert filter_obj.normalize() is None

    def test_model_is_frozen(self) -> None:
        """Verify model is immutable (frozen=True)."""
        filter_obj = ModelPolicyTypeFilter.from_string("test")

        with pytest.raises(ValidationError):
            filter_obj.string_value = "changed"  # type: ignore[misc]


# =============================================================================
# Comparison Tests: Standard Pydantic vs Custom __bool__
# =============================================================================


@pytest.mark.unit
class TestOptionalModelsBoolComparison:
    """Demonstrate the difference between standard Pydantic and custom __bool__.

    These tests explicitly show how optional wrapper models differ from standard
    Pydantic behavior where bool(model) is always True for any valid instance.
    """

    def test_standard_pydantic_always_true(self) -> None:
        """Standard Pydantic models always evaluate to True.

        This test shows the standard behavior that optional models
        intentionally override for more idiomatic conditional checks.
        """

        class StandardModel(BaseModel):
            value: str | None = None

        model = StandardModel(value=None)

        # Standard Pydantic: bool(model) is always True
        assert bool(model) is True

    def test_optional_models_can_be_false(self) -> None:
        """Optional wrapper models can evaluate to False when appropriate.

        This is the key difference from standard Pydantic behavior.
        """
        # All these are valid model instances, but evaluate to False
        assert bool(ModelOptionalString()) is False
        assert bool(ModelOptionalUUID()) is False
        assert bool(ModelOptionalCorrelationId.none()) is False
        assert bool(ModelPolicyTypeFilter.none()) is False

    def test_optional_models_can_be_true(self) -> None:
        """Optional wrapper models evaluate to True when value is present."""
        assert bool(ModelOptionalString(value="test")) is True
        assert bool(ModelOptionalUUID(value=uuid4())) is True
        assert bool(ModelOptionalCorrelationId.generate()) is True
        assert bool(ModelPolicyTypeFilter.from_string("test")) is True


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
class TestOptionalModelsEdgeCases:
    """Edge case tests for optional models __bool__ behavior."""

    def test_optional_string_whitespace_is_truthy(self) -> None:
        """Whitespace-only string is still a value, so truthy."""
        opt = ModelOptionalString(value="   ")
        assert bool(opt) is True

    def test_optional_string_newline_is_truthy(self) -> None:
        """Newline string is still a value, so truthy."""
        opt = ModelOptionalString(value="\n")
        assert bool(opt) is True

    def test_policy_filter_with_both_values_uses_enum(self) -> None:
        """When both string and enum are set, enum takes precedence in normalize."""
        filter_obj = ModelPolicyTypeFilter(
            string_value="string_value",
            enum_value=EnumPolicyType.REDUCER,
        )

        # Should have value (truthy)
        assert bool(filter_obj) is True

        # normalize() returns enum value
        assert filter_obj.normalize() == "reducer"

    def test_in_list_comprehension_filter(self) -> None:
        """Demonstrate use in list comprehension filtering."""
        items = [
            ModelOptionalString(value="a"),
            ModelOptionalString(),  # Empty
            ModelOptionalString(value="b"),
            ModelOptionalString(value=None),  # Empty
            ModelOptionalString(value="c"),
        ]

        # Filter only items with values
        with_values = [item for item in items if item]
        assert len(with_values) == 3
        assert all(item.value is not None for item in with_values)

    def test_in_any_all_builtins(self) -> None:
        """Verify __bool__ works with all() and any() built-ins."""
        items = [
            ModelOptionalString(),
            ModelOptionalString(value="present"),
            ModelOptionalString(),
        ]

        # any() should find the one with value
        assert any(items) is True

        # all() should be False since some are empty
        assert all(items) is False

        # all() with all empty should be False
        all_empty = [ModelOptionalString(), ModelOptionalString()]
        assert all(all_empty) is False
        assert any(all_empty) is False

        # all() with all present should be True
        all_present = [ModelOptionalString(value="a"), ModelOptionalString(value="b")]
        assert all(all_present) is True
        assert any(all_present) is True
