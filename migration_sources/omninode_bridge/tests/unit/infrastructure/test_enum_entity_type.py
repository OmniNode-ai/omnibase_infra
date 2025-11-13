#!/usr/bin/env python3
"""Unit tests for EnumEntityType.

Tests cover:
- Enum values and inheritance
- String behavior and serialization
- All enum values and properties
"""

import json
from enum import Enum

import pytest

from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType


class TestEnumEntityType:
    """Test suite for EnumEntityType."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert EnumEntityType.WORKFLOW_EXECUTION == "workflow_execution"
        assert EnumEntityType.WORKFLOW_STEP == "workflow_step"
        assert EnumEntityType.METADATA_STAMP == "metadata_stamp"
        assert EnumEntityType.FSM_TRANSITION == "fsm_transition"
        assert EnumEntityType.BRIDGE_STATE == "bridge_state"
        assert EnumEntityType.NODE_HEARTBEAT == "node_heartbeat"
        assert EnumEntityType.NODE_HEALTH_METRICS == "node_health_metrics"

    def test_enum_inheritance(self):
        """Test that enum inherits from str and Enum."""
        assert issubclass(EnumEntityType, str)
        assert issubclass(EnumEntityType, Enum)

    def test_enum_string_behavior(self):
        """Test that enum behaves like a string."""
        entity_type = EnumEntityType.WORKFLOW_EXECUTION
        assert isinstance(entity_type, str)
        assert entity_type == "workflow_execution"
        assert len(entity_type) == 18

    def test_enum_serialization(self):
        """Test enum serialization to JSON."""
        entity_type = EnumEntityType.METADATA_STAMP
        serialized = entity_type.value
        assert serialized == "metadata_stamp"

        # Test JSON serialization
        json_str = json.dumps(entity_type)
        assert json_str == '"metadata_stamp"'

        # Test JSON deserialization
        deserialized = json.loads(json_str)
        assert deserialized == "metadata_stamp"

    def test_enum_iteration(self):
        """Test that all enum values can be iterated."""
        values = list(EnumEntityType)
        assert len(values) == 7  # Verify we have all expected values

        # Check that specific values are in the iteration
        assert EnumEntityType.WORKFLOW_EXECUTION in values
        assert EnumEntityType.WORKFLOW_STEP in values
        assert EnumEntityType.METADATA_STAMP in values
        assert EnumEntityType.FSM_TRANSITION in values
        assert EnumEntityType.BRIDGE_STATE in values
        assert EnumEntityType.NODE_HEARTBEAT in values
        assert EnumEntityType.NODE_HEALTH_METRICS in values

    def test_enum_membership(self):
        """Test enum membership operations."""
        assert "workflow_execution" in EnumEntityType
        assert "workflow_step" in EnumEntityType
        assert "metadata_stamp" in EnumEntityType
        assert "fsm_transition" in EnumEntityType
        assert "bridge_state" in EnumEntityType
        assert "node_heartbeat" in EnumEntityType
        assert "node_health_metrics" in EnumEntityType

        assert "invalid_entity" not in EnumEntityType

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        entity_type1 = EnumEntityType.WORKFLOW_EXECUTION
        entity_type2 = EnumEntityType.WORKFLOW_EXECUTION
        assert entity_type1 == entity_type2
        assert entity_type1 is entity_type2

        # Test string comparison
        assert entity_type1 == "workflow_execution"
        assert entity_type1 == "workflow_execution"

    def test_enum_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            EnumEntityType("invalid_entity")

    def test_enum_docstring(self):
        """Test that enum has proper docstring."""
        assert EnumEntityType.__doc__ is not None
        assert "Database entity types" in EnumEntityType.__doc__
        assert "EntityRegistry system" in EnumEntityType.__doc__

    def test_enum_value_docstrings(self):
        """Test that enum values have proper docstrings."""
        # Check that the enum class has a docstring
        assert EnumEntityType.__doc__ is not None
        assert "Database entity types" in EnumEntityType.__doc__
        assert "EntityRegistry system" in EnumEntityType.__doc__
