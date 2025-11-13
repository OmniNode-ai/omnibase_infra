#!/usr/bin/env python3
"""Unit tests for FSMStateManager in reducer node.

Tests cover:
- FSM state initialization
- State transition validation
- State persistence and recovery
- Transition history tracking
- Error handling
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

# Import the module to test
from omninode_bridge.nodes.reducer.v1_0_0.fsm_state_manager import FSMStateManager


class TestFSMStateManager:
    """Test suite for FSMStateManager."""

    @pytest.fixture
    def mock_container(self):
        """Create a mock ONEX container."""
        container = MagicMock()
        return container

    @pytest.fixture
    def mock_fsm_config(self):
        """Create a mock FSM configuration."""
        config = MagicMock()
        # Create mock states
        state1 = MagicMock()
        state1.state_name = "pending"
        state2 = MagicMock()
        state2.state_name = "processing"
        state3 = MagicMock()
        state3.state_name = "completed"
        state4 = MagicMock()
        state4.state_name = "failed"

        config.states = [state1, state2, state3, state4]

        # Create mock transitions
        transition1 = MagicMock()
        transition1.from_state = "pending"
        transition1.to_state = "processing"
        transition1.trigger = "start"

        transition2 = MagicMock()
        transition2.from_state = "processing"
        transition2.to_state = "completed"
        transition2.trigger = "complete"

        transition3 = MagicMock()
        transition3.from_state = "processing"
        transition3.to_state = "failed"
        transition3.trigger = "error"

        config.transitions = [transition1, transition2, transition3]

        return config

    @pytest.fixture
    def state_manager(self, mock_container, mock_fsm_config):
        """Create FSMStateManager instance for testing."""
        return FSMStateManager(mock_container, mock_fsm_config)

    def test_init_with_fsm_config(self, mock_container, mock_fsm_config):
        """Test initialization with FSM configuration."""
        manager = FSMStateManager(mock_container, mock_fsm_config)

        assert manager.container == mock_container
        assert manager._fsm_config == mock_fsm_config
        assert manager._state_cache == {}
        assert manager._transition_history == {}
        assert manager._state_timestamps == {}
        # States are normalized to uppercase for consistency
        assert manager._valid_states == {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}

    def test_init_without_fsm_config(self, mock_container):
        """Test initialization without FSM configuration - uses fallback states."""
        manager = FSMStateManager(mock_container)

        assert manager.container == mock_container
        assert manager._fsm_config is None
        assert manager._state_cache == {}
        assert manager._transition_history == {}
        assert manager._state_timestamps == {}
        # Fallback states should be present when no FSM config provided
        assert hasattr(manager, "_valid_states")
        assert manager._valid_states == {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}

    @pytest.mark.asyncio
    async def test_initialize_state_new_workflow(self, state_manager):
        """Test initializing state for a new workflow."""
        workflow_id = uuid4()
        initial_state = "pending"

        result = await state_manager.initialize_workflow(workflow_id, initial_state)

        assert result is True
        assert workflow_id in state_manager._state_cache
        # States are normalized to uppercase
        assert (
            state_manager._state_cache[workflow_id]["current_state"]
            == initial_state.upper()
        )
        assert state_manager._state_cache[workflow_id]["previous_state"] is None
        assert state_manager._state_cache[workflow_id]["transition_count"] == 0
        assert workflow_id in state_manager._state_timestamps

    @pytest.mark.asyncio
    async def test_initialize_state_existing_workflow(self, state_manager):
        """Test initializing state for an existing workflow."""
        workflow_id = uuid4()
        initial_state = "pending"

        # Initialize once
        await state_manager.initialize_workflow(workflow_id, initial_state)

        # Try to initialize again
        result = await state_manager.initialize_workflow(workflow_id, "processing")

        assert result is False
        # State should still be the original (uppercase)
        assert (
            state_manager._state_cache[workflow_id]["current_state"]
            == initial_state.upper()
        )

    @pytest.mark.asyncio
    async def test_initialize_state_invalid_state(self, state_manager):
        """Test initializing with an invalid state."""
        workflow_id = uuid4()
        invalid_state = "invalid_state"

        result = await state_manager.initialize_workflow(workflow_id, invalid_state)

        assert result is False
        assert workflow_id not in state_manager._state_cache

    @pytest.mark.asyncio
    async def test_transition_state_valid(self, state_manager):
        """Test valid state transition."""
        workflow_id = uuid4()

        # Initialize state
        await state_manager.initialize_workflow(workflow_id, "pending")

        # Make valid transition
        result = await state_manager.transition_state(
            workflow_id, "pending", "processing", "start"
        )

        assert result is True
        # States normalized to uppercase
        assert state_manager._state_cache[workflow_id]["current_state"] == "PROCESSING"
        assert state_manager._state_cache[workflow_id]["previous_state"] == "PENDING"
        assert state_manager._state_cache[workflow_id]["transition_count"] == 1
        assert len(state_manager._transition_history[workflow_id]) == 1

    @pytest.mark.asyncio
    async def test_transition_state_invalid(self, state_manager):
        """Test invalid state transition."""
        workflow_id = uuid4()

        # Initialize state
        await state_manager.initialize_workflow(workflow_id, "pending")

        # Try invalid transition
        result = await state_manager.transition_state(
            workflow_id, "pending", "completed", "skip"
        )

        assert result is False
        # States are normalized to uppercase
        assert state_manager._state_cache[workflow_id]["current_state"] == "PENDING"
        assert state_manager._state_cache[workflow_id]["transition_count"] == 0

    @pytest.mark.asyncio
    async def test_transition_state_nonexistent_workflow(self, state_manager):
        """Test transition for non-existent workflow."""
        workflow_id = uuid4()

        # Try transition without initialization
        result = await state_manager.transition_state(
            workflow_id, "pending", "processing", "start"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_transition_state_current_state_mismatch(self, state_manager):
        """Test transition with current state mismatch."""
        workflow_id = uuid4()

        # Initialize state
        await state_manager.initialize_workflow(workflow_id, "pending")

        # Try transition with wrong current state
        result = await state_manager.transition_state(
            workflow_id, "processing", "completed", "complete"
        )

        assert result is False
        # States are normalized to uppercase
        assert state_manager._state_cache[workflow_id]["current_state"] == "PENDING"

    @pytest.mark.asyncio
    async def test_transition_state_with_metadata(self, state_manager):
        """Test state transition with metadata."""
        workflow_id = uuid4()
        metadata = {"reason": "test", "user": "test_user"}

        # Initialize state
        await state_manager.initialize_workflow(workflow_id, "pending")

        # Make transition with metadata
        result = await state_manager.transition_state(
            workflow_id, "pending", "processing", "start", metadata
        )

        assert result is True
        transition_record = state_manager._transition_history[workflow_id][0]
        assert transition_record["metadata"] == metadata

    def test_get_state_existing(self, state_manager):
        """Test getting state for existing workflow."""
        workflow_id = uuid4()

        # Manually add state to cache
        state_manager._state_cache[workflow_id] = {"current_state": "processing"}

        state = state_manager.get_state(workflow_id)
        assert state == "processing"

    def test_get_state_nonexistent(self, state_manager):
        """Test getting state for non-existent workflow."""
        workflow_id = uuid4()

        state = state_manager.get_state(workflow_id)
        assert state is None

    def test_get_transition_history_existing(self, state_manager):
        """Test getting transition history for existing workflow."""
        workflow_id = uuid4()

        # Manually add transition history
        transition1 = {"from_state": "pending", "to_state": "processing"}
        transition2 = {"from_state": "processing", "to_state": "completed"}
        state_manager._transition_history[workflow_id] = [transition1, transition2]

        history = state_manager.get_transition_history(workflow_id)
        assert len(history) == 2
        assert history[0] == transition1
        assert history[1] == transition2

    def test_get_transition_history_nonexistent(self, state_manager):
        """Test getting transition history for non-existent workflow."""
        workflow_id = uuid4()

        history = state_manager.get_transition_history(workflow_id)
        assert history == []

    @pytest.mark.asyncio
    async def test_recover_states(self, state_manager):
        """Test state recovery stub implementation."""
        result = await state_manager.recover_states()

        # Verify return value structure (stub implementation)
        assert "recovered" in result
        assert "failed" in result
        assert "total" in result
        assert result["recovered"] == 0
        assert result["failed"] == 0
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_persist_state_transition(self, state_manager):
        """Test persist state transition stub."""
        workflow_id = uuid4()

        # Initialize workflow first so state_cache has data for the method
        await state_manager.initialize_workflow(workflow_id, "pending")

        transition_record = {
            "from_state": "PENDING",
            "to_state": "PROCESSING",
            "trigger": "start",
            "timestamp": datetime.now(UTC),
            "metadata": {},
        }

        # Test the stub implementation (currently a pass statement)
        # This method is a placeholder for future PostgreSQL persistence
        await state_manager._persist_state_transition(workflow_id, transition_record)

        # No error should be raised - stub implementation is a no-op
        assert True

    def test_validate_transition_invalid_from_state(self, state_manager):
        """Test _validate_transition with invalid from_state."""
        # Test with invalid from_state
        result = state_manager._validate_transition("INVALID_STATE", "PROCESSING")
        assert result is False

    def test_validate_transition_invalid_to_state(self, state_manager):
        """Test _validate_transition with invalid to_state."""
        # Test with invalid to_state
        result = state_manager._validate_transition("PENDING", "INVALID_STATE")
        assert result is False

    def test_validate_transition_both_invalid(self, state_manager):
        """Test _validate_transition with both states invalid."""
        # Test with both invalid
        result = state_manager._validate_transition("INVALID_FROM", "INVALID_TO")
        assert result is False

    def test_validate_transition_valid(self, state_manager):
        """Test _validate_transition with valid transition."""
        # Test valid transition
        result = state_manager._validate_transition("PENDING", "PROCESSING")
        assert result is True

    def test_validate_transition_not_allowed(self, state_manager):
        """Test _validate_transition with valid states but not allowed transition."""
        # Test states exist but transition not allowed
        result = state_manager._validate_transition("COMPLETED", "PENDING")
        assert result is False
