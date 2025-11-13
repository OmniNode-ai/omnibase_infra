#!/usr/bin/env python3
"""Unit tests for EnumAggregationType.

Tests cover:
- Enum value definitions
- String representation
- Aggregation strategy validation
- Enum comparison
"""

import pytest

from omninode_bridge.nodes.reducer.v1_0_0.models.enum_aggregation_type import (
    EnumAggregationType,
)


class TestEnumAggregationType:
    """Test suite for EnumAggregationType."""

    def test_enum_values(self):
        """Test that all expected enum values are defined."""
        assert EnumAggregationType.NAMESPACE_GROUPING.value == "namespace_grouping"
        assert EnumAggregationType.TIME_WINDOW.value == "time_window"
        assert EnumAggregationType.FILE_TYPE_GROUPING.value == "file_type_grouping"
        assert EnumAggregationType.SIZE_BUCKETS.value == "size_buckets"
        assert EnumAggregationType.WORKFLOW_GROUPING.value == "workflow_grouping"
        assert EnumAggregationType.CUSTOM.value == "custom"

    def test_enum_inheritance(self):
        """Test enum inherits from str."""
        assert isinstance(EnumAggregationType.NAMESPACE_GROUPING, str)

    def test_enum_string_behavior(self):
        """Test enum behaves like string."""
        assert str(EnumAggregationType.NAMESPACE_GROUPING) == "namespace_grouping"
        assert EnumAggregationType.NAMESPACE_GROUPING == "namespace_grouping"

    def test_enum_str_method(self):
        """Test __str__ method returns enum value."""
        # Test all enum values
        assert str(EnumAggregationType.NAMESPACE_GROUPING) == "namespace_grouping"
        assert str(EnumAggregationType.TIME_WINDOW) == "time_window"
        assert str(EnumAggregationType.FILE_TYPE_GROUPING) == "file_type_grouping"
        assert str(EnumAggregationType.SIZE_BUCKETS) == "size_buckets"
        assert str(EnumAggregationType.WORKFLOW_GROUPING) == "workflow_grouping"
        assert str(EnumAggregationType.CUSTOM) == "custom"

    def test_enum_iteration(self):
        """Test enum iteration."""
        all_types = list(EnumAggregationType)
        assert len(all_types) == 6  # Total number of aggregation types
        assert EnumAggregationType.NAMESPACE_GROUPING in all_types
        assert EnumAggregationType.TIME_WINDOW in all_types
        assert EnumAggregationType.CUSTOM in all_types

    def test_enum_membership(self):
        """Test enum membership."""
        assert EnumAggregationType.NAMESPACE_GROUPING == "namespace_grouping"
        assert EnumAggregationType.NAMESPACE_GROUPING != "invalid_type"

    def test_enum_serialization(self):
        """Test enum serialization to string."""
        agg_type = EnumAggregationType.NAMESPACE_GROUPING
        # Can be serialized to string
        serialized = str(agg_type)
        assert serialized == "namespace_grouping"

        # Can be compared with string
        assert agg_type == serialized

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        type1 = EnumAggregationType.NAMESPACE_GROUPING
        type2 = EnumAggregationType.NAMESPACE_GROUPING
        type3 = EnumAggregationType.TIME_WINDOW

        # Same types are equal
        assert type1 == type2
        assert type1 is type2  # Same instance

        # Different types are not equal
        assert type1 != type3

    def test_enum_invalid_value(self):
        """Test accessing invalid enum value raises error."""
        with pytest.raises(ValueError):
            EnumAggregationType("invalid_aggregation_type")

    def test_enum_docstring(self):
        """Test enum has docstring."""
        assert EnumAggregationType.__doc__ is not None
        assert "Aggregation strategy types" in EnumAggregationType.__doc__

    def test_enum_value_docstrings(self):
        """Test enum values have docstrings."""
        # Check a few key types have docstrings
        assert EnumAggregationType.NAMESPACE_GROUPING.__doc__ is not None
        assert EnumAggregationType.TIME_WINDOW.__doc__ is not None
        assert EnumAggregationType.CUSTOM.__doc__ is not None

    def test_all_aggregation_types_covered(self):
        """Test that all documented aggregation types are present."""
        expected_types = {
            "namespace_grouping",
            "time_window",
            "file_type_grouping",
            "size_buckets",
            "workflow_grouping",
            "custom",
        }
        actual_types = {agg_type.value for agg_type in EnumAggregationType}
        assert actual_types == expected_types
