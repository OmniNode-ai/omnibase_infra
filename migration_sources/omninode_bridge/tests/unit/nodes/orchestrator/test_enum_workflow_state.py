#!/usr/bin/env python3
"""Unit tests for EnumWorkflowState.

Tests cover:
- Enum values and inheritance
- String behavior and serialization
- Terminal state detection
- State transition validation
- All enum values and properties
"""

import json
from enum import Enum

import pytest

from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state import (
    EnumWorkflowState,
)


class TestEnumWorkflowState:
    """Test suite for EnumWorkflowState."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert EnumWorkflowState.PENDING == "pending"
        assert EnumWorkflowState.PROCESSING == "processing"
        assert EnumWorkflowState.COMPLETED == "completed"
        assert EnumWorkflowState.FAILED == "failed"

    def test_enum_inheritance(self):
        """Test that enum inherits from str and Enum."""
        assert issubclass(EnumWorkflowState, str)
        assert issubclass(EnumWorkflowState, Enum)

    def test_enum_string_behavior(self):
        """Test that enum behaves like a string."""
        state = EnumWorkflowState.PENDING
        assert isinstance(state, str)
        assert state == "pending"
        assert len(state) == 7

    def test_enum_serialization(self):
        """Test enum serialization to JSON."""
        state = EnumWorkflowState.PROCESSING
        serialized = state.value
        assert serialized == "processing"

        # Test JSON serialization
        json_str = json.dumps(state)
        assert json_str == '"processing"'

        # Test JSON deserialization
        deserialized = json.loads(json_str)
        assert deserialized == "processing"

    def test_enum_iteration(self):
        """Test that all enum values can be iterated."""
        values = list(EnumWorkflowState)
        assert len(values) == 4  # Verify we have all expected values

        # Check that specific values are in the iteration
        assert EnumWorkflowState.PENDING in values
        assert EnumWorkflowState.PROCESSING in values
        assert EnumWorkflowState.COMPLETED in values
        assert EnumWorkflowState.FAILED in values

    def test_enum_membership(self):
        """Test enum membership operations."""
        assert "pending" in EnumWorkflowState
        assert "processing" in EnumWorkflowState
        assert "completed" in EnumWorkflowState
        assert "failed" in EnumWorkflowState

        assert "invalid_state" not in EnumWorkflowState

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        state1 = EnumWorkflowState.PENDING
        state2 = EnumWorkflowState.PENDING
        assert state1 == state2
        assert state1 is state2

        # Test string comparison
        assert state1 == "pending"
        assert state1 == "pending"

    def test_enum_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            EnumWorkflowState("invalid_state")

    def test_is_terminal(self):
        """Test terminal state detection."""
        # Terminal states
        assert EnumWorkflowState.COMPLETED.is_terminal() is True
        assert EnumWorkflowState.FAILED.is_terminal() is True

        # Non-terminal states
        assert EnumWorkflowState.PENDING.is_terminal() is False
        assert EnumWorkflowState.PROCESSING.is_terminal() is False

    def test_can_transition_to_valid_transitions(self):
        """Test valid state transitions."""
        # PENDING can transition to PROCESSING or FAILED
        assert (
            EnumWorkflowState.PENDING.can_transition_to(EnumWorkflowState.PROCESSING)
            is True
        )
        assert (
            EnumWorkflowState.PENDING.can_transition_to(EnumWorkflowState.FAILED)
            is True
        )

        # PROCESSING can transition to COMPLETED or FAILED
        assert (
            EnumWorkflowState.PROCESSING.can_transition_to(EnumWorkflowState.COMPLETED)
            is True
        )
        assert (
            EnumWorkflowState.PROCESSING.can_transition_to(EnumWorkflowState.FAILED)
            is True
        )

    def test_can_transition_to_invalid_transitions(self):
        """Test invalid state transitions."""
        # PENDING cannot transition to COMPLETED
        assert (
            EnumWorkflowState.PENDING.can_transition_to(EnumWorkflowState.COMPLETED)
            is False
        )

        # PROCESSING cannot transition to PENDING
        assert (
            EnumWorkflowState.PROCESSING.can_transition_to(EnumWorkflowState.PENDING)
            is False
        )

        # COMPLETED cannot transition to any other state
        assert (
            EnumWorkflowState.COMPLETED.can_transition_to(EnumWorkflowState.PENDING)
            is False
        )
        assert (
            EnumWorkflowState.COMPLETED.can_transition_to(EnumWorkflowState.PROCESSING)
            is False
        )
        assert (
            EnumWorkflowState.COMPLETED.can_transition_to(EnumWorkflowState.FAILED)
            is False
        )

        # FAILED cannot transition to any other state
        assert (
            EnumWorkflowState.FAILED.can_transition_to(EnumWorkflowState.PENDING)
            is False
        )
        assert (
            EnumWorkflowState.FAILED.can_transition_to(EnumWorkflowState.PROCESSING)
            is False
        )
        assert (
            EnumWorkflowState.FAILED.can_transition_to(EnumWorkflowState.COMPLETED)
            is False
        )

    def test_can_transition_to_self(self):
        """Test that states cannot transition to themselves."""
        assert (
            EnumWorkflowState.PENDING.can_transition_to(EnumWorkflowState.PENDING)
            is False
        )
        assert (
            EnumWorkflowState.PROCESSING.can_transition_to(EnumWorkflowState.PROCESSING)
            is False
        )
        assert (
            EnumWorkflowState.COMPLETED.can_transition_to(EnumWorkflowState.COMPLETED)
            is False
        )
        assert (
            EnumWorkflowState.FAILED.can_transition_to(EnumWorkflowState.FAILED)
            is False
        )

    def test_all_possible_transitions(self):
        """Test all possible state transitions matrix."""
        # Create a matrix of all possible transitions
        all_states = list(EnumWorkflowState)
        transition_matrix = {}

        for from_state in all_states:
            transition_matrix[from_state] = {}
            for to_state in all_states:
                transition_matrix[from_state][to_state] = from_state.can_transition_to(
                    to_state
                )

        # Verify specific transitions
        assert (
            transition_matrix[EnumWorkflowState.PENDING][EnumWorkflowState.PROCESSING]
            is True
        )
        assert (
            transition_matrix[EnumWorkflowState.PENDING][EnumWorkflowState.FAILED]
            is True
        )
        assert (
            transition_matrix[EnumWorkflowState.PENDING][EnumWorkflowState.COMPLETED]
            is False
        )

        assert (
            transition_matrix[EnumWorkflowState.PROCESSING][EnumWorkflowState.COMPLETED]
            is True
        )
        assert (
            transition_matrix[EnumWorkflowState.PROCESSING][EnumWorkflowState.FAILED]
            is True
        )
        assert (
            transition_matrix[EnumWorkflowState.PROCESSING][EnumWorkflowState.PENDING]
            is False
        )

        # Terminal states have no outgoing transitions
        for to_state in all_states:
            assert transition_matrix[EnumWorkflowState.COMPLETED][to_state] is False
            assert transition_matrix[EnumWorkflowState.FAILED][to_state] is False

    def test_enum_docstring(self):
        """Test that enum has proper docstring."""
        assert EnumWorkflowState.__doc__ is not None
        assert "Finite State Machine" in EnumWorkflowState.__doc__

    def test_enum_value_docstrings(self):
        """Test that enum class has comprehensive documentation.

        Note: Python enums expose the class docstring via __doc__, not individual
        value docstrings. The value docstrings exist in source for developer reference.
        """
        # Class docstring should describe all states
        class_doc = EnumWorkflowState.__doc__
        assert class_doc is not None
        assert "PENDING" in class_doc
        assert "PROCESSING" in class_doc
        assert "COMPLETED" in class_doc
        assert "FAILED" in class_doc
