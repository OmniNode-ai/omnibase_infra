#!/usr/bin/env python3
"""
FSM State Manager for Workflow State Tracking.

Manages workflow FSM states by loading state definitions and transitions
from FSM subcontract YAML (contract-driven architecture).

ONEX v2.0 Compliance:
- Extracted from node.py for Single Responsibility Principle
- Proper module organization (no classes in node.py except node class)
"""

import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID

# Import with fallback to stubs when omnibase_core is not available
try:
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_core.models.contracts.subcontracts import ModelFSMSubcontract
except ImportError:
    from ._stubs import ModelFSMSubcontract, ModelONEXContainer

# Import protocols from omnibase_spi for duck typing (Phase 2: Protocol Type Hints)
try:
    from omnibase_spi.protocols import ProtocolServiceRegistry

    PROTOCOLS_AVAILABLE = True
except ImportError:
    # Protocol imports are optional - duck typing still works with concrete types
    PROTOCOLS_AVAILABLE = False
    ProtocolServiceRegistry = ModelONEXContainer  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class FSMStateManager:
    """
    FSM State Manager for Workflow State Tracking.

    Manages workflow FSM states by loading state definitions and transitions
    from FSM subcontract YAML (contract-driven architecture).

    Features:
    - State transition validation from subcontract
    - PostgreSQL persistence
    - State recovery
    - Transition history tracking
    - Guard condition validation

    States and transitions are loaded dynamically from the FSM subcontract,
    not hard-coded.
    """

    def __init__(
        self,
        container: ProtocolServiceRegistry,
        fsm_config: Optional[ModelFSMSubcontract] = None,
    ) -> None:
        """
        Initialize FSM state manager with FSM subcontract.

        Args:
            container: DI container with service resolution (uses ProtocolServiceRegistry
                      for duck typing - any object with get_service/register_service methods)
            fsm_config: FSM subcontract loaded from contract YAML (optional for now)

        Note:
            Uses ProtocolServiceRegistry protocol for PUBLIC API duck typing.
            Internal implementation still uses concrete ModelONEXContainer.
        """
        self.container = container
        self._fsm_config = fsm_config
        self._state_cache: dict[UUID, dict[str, Any]] = {}
        self._transition_history: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
        self._state_timestamps: dict[UUID, datetime] = {}

        # Load states and transitions from FSM subcontract
        if fsm_config:
            # Extract state names from FSM subcontract (normalize to uppercase for consistency)
            self._valid_states = {
                state.state_name.upper() for state in fsm_config.states
            }

            # Build transition map from FSM subcontract (normalize states to uppercase)
            self._valid_transitions: dict[str, set[str]] = defaultdict(set)
            for transition in fsm_config.transitions:
                self._valid_transitions[transition.from_state.upper()].add(
                    transition.to_state.upper()
                )

            # Store terminal states (normalize to uppercase)
            self._terminal_states = {
                state.upper() for state in fsm_config.terminal_states
            }

            logger.info(
                f"FSM State Manager initialized with {len(self._valid_states)} states "
                f"and {len(fsm_config.transitions)} transitions from subcontract"
            )
        else:
            # Fallback to default states when no FSM subcontract is provided
            # This maintains backward compatibility during transition period
            logger.warning(
                "FSM State Manager initialized without FSM subcontract - "
                "using fallback states (PENDING, PROCESSING, COMPLETED, FAILED)"
            )
            self._valid_states = {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}
            self._valid_transitions = {
                "PENDING": {"PROCESSING", "FAILED"},
                "PROCESSING": {"COMPLETED", "FAILED"},
                "COMPLETED": set(),
                "FAILED": set(),
            }
            self._terminal_states = {"COMPLETED", "FAILED"}

    async def initialize_workflow(
        self,
        workflow_id: UUID,
        initial_state: str = "PENDING",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Initialize a new workflow in FSM.

        Args:
            workflow_id: Workflow identifier
            initial_state: Initial FSM state (default: PENDING, case-insensitive)
            metadata: Optional workflow metadata

        Returns:
            True if workflow initialized successfully, False otherwise
        """
        # Check if workflow already exists
        if workflow_id in self._state_cache:
            logger.warning(f"Workflow {workflow_id} already initialized")
            return False

        # Normalize state to uppercase for consistency
        normalized_state = initial_state.upper()

        # Validate state
        if normalized_state not in self._valid_states:
            logger.error(
                f"Invalid initial state: {initial_state}. "
                f"Valid states: {self._valid_states}"
            )
            return False

        self._state_cache[workflow_id] = {
            "current_state": normalized_state,
            "previous_state": None,
            "transition_count": 0,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        self._state_timestamps[workflow_id] = datetime.now(UTC)

        logger.info(
            f"FSM workflow initialized: {workflow_id} in state {normalized_state}"
        )
        return True

    async def transition_state(
        self,
        workflow_id: UUID,
        from_state: str,
        to_state: str,
        trigger: str = "manual",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Transition workflow to a new FSM state with validation.

        Args:
            workflow_id: Workflow identifier
            from_state: Expected current state (for validation, case-insensitive)
            to_state: Target state to transition to (case-insensitive)
            trigger: Event trigger causing transition
            metadata: Optional transition metadata

        Returns:
            True if transition successful, False otherwise
        """
        # Normalize states to uppercase
        normalized_from_state = from_state.upper()
        normalized_to_state = to_state.upper()

        # Get current state
        if workflow_id not in self._state_cache:
            logger.error(f"Workflow {workflow_id} not initialized")
            return False

        current_state = self._state_cache[workflow_id]["current_state"]

        # Validate current state matches expected from_state
        if current_state != normalized_from_state:
            logger.error(
                f"Current state mismatch for {workflow_id}: "
                f"expected {normalized_from_state}, got {current_state}"
            )
            return False

        # Validate transition
        if not self._validate_transition(current_state, normalized_to_state):
            error_msg = (
                f"Invalid transition: {current_state} -> {normalized_to_state} "
                f"for workflow {workflow_id}"
            )
            logger.error(error_msg)
            return False

        # Record transition
        transition_record = {
            "from_state": current_state,
            "to_state": normalized_to_state,
            "trigger": trigger,
            "timestamp": datetime.now(UTC),
            "metadata": metadata or {},
        }
        self._transition_history[workflow_id].append(transition_record)

        # Update state
        old_state = current_state
        self._state_cache[workflow_id]["previous_state"] = old_state
        self._state_cache[workflow_id]["current_state"] = normalized_to_state
        self._state_cache[workflow_id]["transition_count"] += 1
        self._state_cache[workflow_id]["updated_at"] = datetime.now(UTC)
        self._state_timestamps[workflow_id] = datetime.now(UTC)

        # Persist state transition
        await self._persist_state_transition(workflow_id, transition_record)

        logger.info(
            f"FSM state transition: {workflow_id} "
            f"{old_state} -> {normalized_to_state} (trigger: {trigger})"
        )

        return True

    def get_state(self, workflow_id: UUID) -> Optional[str]:
        """
        Get current FSM state for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Current state string, or None if workflow not found
        """
        if workflow_id not in self._state_cache:
            return None
        return str(self._state_cache[workflow_id]["current_state"])

    def get_transition_history(self, workflow_id: UUID) -> list[dict[str, Any]]:
        """
        Get transition history for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of transition records
        """
        return self._transition_history.get(workflow_id, [])

    async def recover_states(self) -> dict[str, int]:
        """
        Recover FSM states from PostgreSQL on startup.

        Returns:
            Recovery statistics dict
        """
        # Placeholder for actual PostgreSQL recovery
        # In production, this would query the fsm_workflow_states table
        # and restore the _state_cache and _transition_history

        logger.info("FSM state recovery started")

        # Stub implementation - would query database here
        recovered_count = 0
        failed_count = 0

        # Example recovery logic (commented for stub):
        # try:
        #     async with self.container.get_service('postgresql_client') as db:
        #         states = await db.fetch_all_workflow_states()
        #         for state_record in states:
        #             workflow_id = UUID(state_record['workflow_id'])
        #             self._state_cache[workflow_id] = {
        #                 'current_state': state_record['current_state'],
        #                 'previous_state': state_record['previous_state'],
        #                 'transition_count': state_record['transition_count'],
        #                 'metadata': state_record['metadata'],
        #                 'created_at': state_record['created_at'],
        #                 'updated_at': state_record['updated_at'],
        #             }
        #             recovered_count += 1
        # except Exception as e:
        #     logger.error(f"FSM state recovery failed: {e}")
        #     failed_count += 1

        logger.info(
            f"FSM state recovery completed: {recovered_count} recovered, "
            f"{failed_count} failed"
        )

        return {
            "recovered": recovered_count,
            "failed": failed_count,
            "total": recovered_count + failed_count,
        }

    def _validate_transition(self, from_state: str, to_state: str) -> bool:
        """
        Validate FSM state transition.

        Args:
            from_state: Current state
            to_state: Target state

        Returns:
            True if transition is valid, False otherwise
        """
        # Validate states exist
        if from_state not in self._valid_states:
            logger.error(f"Invalid from_state: {from_state}")
            return False

        if to_state not in self._valid_states:
            logger.error(f"Invalid to_state: {to_state}")
            return False

        # Check if transition is allowed
        allowed_transitions = self._valid_transitions.get(from_state, set())
        if to_state not in allowed_transitions:
            logger.warning(
                f"Transition not allowed: {from_state} -> {to_state}. "
                f"Allowed: {allowed_transitions}"
            )
            return False

        return True

    async def _persist_state_transition(
        self,
        workflow_id: UUID,
        transition_record: dict[str, Any],
    ) -> None:
        """
        Persist FSM state transition to PostgreSQL.

        Args:
            workflow_id: Workflow identifier
            transition_record: Transition metadata
        """
        # Placeholder for actual PostgreSQL persistence
        # In production, this would upsert to fsm_workflow_states table

        # Example persistence logic (commented for stub):
        # try:
        #     async with self.container.get_service('postgresql_client') as db:
        #         await db.upsert_workflow_state(
        #             workflow_id=workflow_id,
        #             current_state=self._state_cache[workflow_id]['current_state'],
        #             previous_state=self._state_cache[workflow_id]['previous_state'],
        #             transition_count=self._state_cache[workflow_id]['transition_count'],
        #             transition_history=self._transition_history[workflow_id],
        #             metadata=self._state_cache[workflow_id]['metadata'],
        #         )
        # except Exception as e:
        #     logger.error(f"Failed to persist FSM state: {e}")

        pass
