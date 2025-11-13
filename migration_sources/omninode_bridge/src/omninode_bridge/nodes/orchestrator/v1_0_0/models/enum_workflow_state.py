#!/usr/bin/env python3
"""
Workflow State Enumeration for NodeBridgeOrchestrator.

Defines FSM states for metadata stamping workflow orchestration.
Part of the O.N.E. v0.1 compliant bridge architecture.

ONEX v2.0 Compliance:
- Enum-based naming: EnumWorkflowState
- FSM state management for workflow tracking
- Integration with ModelFSMSubcontract
"""

from enum import Enum


class EnumWorkflowState(str, Enum):
    """
    Finite State Machine states for stamping workflows.

    State Transitions:
    - PENDING → PROCESSING: Workflow execution begins
    - PENDING → FAILED: Validation error before processing
    - PROCESSING → COMPLETED: Successful workflow completion
    - PROCESSING → FAILED: Workflow execution error
    - COMPLETED/FAILED: Terminal states (no transitions)

    Usage:
        Used by NodeBridgeOrchestrator to track workflow execution state
        through FSM-based state management with ModelFSMSubcontract.
    """

    PENDING = "pending"
    """Initial state when workflow is queued but not yet executing."""

    PROCESSING = "processing"
    """Active state during workflow execution."""

    COMPLETED = "completed"
    """Terminal state indicating successful workflow completion."""

    FAILED = "failed"
    """Terminal state indicating workflow execution failure."""

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (no further transitions)."""
        return self in (EnumWorkflowState.COMPLETED, EnumWorkflowState.FAILED)

    def can_transition_to(self, target: "EnumWorkflowState") -> bool:
        """
        Validate if transition to target state is allowed.

        Args:
            target: Target state to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        valid_transitions = {
            EnumWorkflowState.PENDING: {
                EnumWorkflowState.PROCESSING,
                EnumWorkflowState.FAILED,  # Allow validation errors
            },
            EnumWorkflowState.PROCESSING: {
                EnumWorkflowState.COMPLETED,
                EnumWorkflowState.FAILED,
            },
            EnumWorkflowState.COMPLETED: set(),  # Terminal state
            EnumWorkflowState.FAILED: set(),  # Terminal state
        }

        return target in valid_transitions.get(self, set())
