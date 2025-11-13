#!/usr/bin/env python3
"""
FSM State Tracker for Code Generation Workflows.

Tracks workflow state transitions through the 6-phase generation process:
1. PENDING → Initial state
2. ANALYZING → Intelligence gathering
3. GENERATING → Code generation
4. VALIDATING → Quality validation
5. COMPLETED → Successful completion
6. FAILED → Failure at any stage

ONEX v2.0 Compliance:
- Suffix-based naming: FSMStateTracker
- Contract-driven state definitions
- Event-driven state transitions
- PostgreSQL persistence for state history
- Pure function pattern with intent emission

Features:
- State transition validation
- Guard condition checking
- Transition history tracking
- State recovery mechanisms
- Intent-based persistence (no direct I/O)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# === Enums ===


class EnumWorkflowState(str, Enum):
    """Workflow FSM states for code generation."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class EnumTransitionEvent(str, Enum):
    """State transition events."""

    START_ANALYSIS = "start_analysis"
    START_GENERATION = "start_generation"
    START_VALIDATION = "start_validation"
    COMPLETE_WORKFLOW = "complete_workflow"
    FAIL_WORKFLOW = "fail_workflow"
    RETRY_WORKFLOW = "retry_workflow"


class EnumIntentType(str, Enum):
    """Intent types for side effects."""

    PERSIST_STATE = "persist_state"
    EMIT_EVENT = "emit_event"
    LOG_TRANSITION = "log_transition"


# === Models ===


@dataclass
class StateTransition:
    """State transition record."""

    transition_id: UUID = field(default_factory=uuid4)
    workflow_id: UUID = field(default_factory=uuid4)
    from_state: EnumWorkflowState = EnumWorkflowState.PENDING
    to_state: EnumWorkflowState = EnumWorkflowState.PENDING
    event: EnumTransitionEvent = EnumTransitionEvent.START_ANALYSIS
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    guard_conditions_met: bool = True
    reason: Optional[str] = None


@dataclass
class WorkflowStateRecord:
    """Workflow state record."""

    workflow_id: UUID
    current_state: EnumWorkflowState
    previous_state: Optional[EnumWorkflowState] = None
    state_entry_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    transition_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelStateIntent(BaseModel):
    """Intent for state persistence."""

    intent_type: EnumIntentType
    workflow_id: UUID
    state: EnumWorkflowState
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# === FSM Configuration ===


@dataclass
class FSMConfiguration:
    """FSM configuration with valid states and transitions."""

    # Valid states
    valid_states: set[EnumWorkflowState] = field(
        default_factory=lambda: {
            EnumWorkflowState.PENDING,
            EnumWorkflowState.ANALYZING,
            EnumWorkflowState.GENERATING,
            EnumWorkflowState.VALIDATING,
            EnumWorkflowState.COMPLETED,
            EnumWorkflowState.FAILED,
        }
    )

    # Terminal states (no further transitions)
    terminal_states: set[EnumWorkflowState] = field(
        default_factory=lambda: {
            EnumWorkflowState.COMPLETED,
            EnumWorkflowState.FAILED,
        }
    )

    # Valid transitions map: from_state -> {to_state: event}
    valid_transitions: dict[
        EnumWorkflowState, dict[EnumWorkflowState, EnumTransitionEvent]
    ] = field(
        default_factory=lambda: {
            EnumWorkflowState.PENDING: {
                EnumWorkflowState.ANALYZING: EnumTransitionEvent.START_ANALYSIS,
                EnumWorkflowState.FAILED: EnumTransitionEvent.FAIL_WORKFLOW,
            },
            EnumWorkflowState.ANALYZING: {
                EnumWorkflowState.GENERATING: EnumTransitionEvent.START_GENERATION,
                EnumWorkflowState.FAILED: EnumTransitionEvent.FAIL_WORKFLOW,
            },
            EnumWorkflowState.GENERATING: {
                EnumWorkflowState.VALIDATING: EnumTransitionEvent.START_VALIDATION,
                EnumWorkflowState.FAILED: EnumTransitionEvent.FAIL_WORKFLOW,
            },
            EnumWorkflowState.VALIDATING: {
                EnumWorkflowState.COMPLETED: EnumTransitionEvent.COMPLETE_WORKFLOW,
                EnumWorkflowState.FAILED: EnumTransitionEvent.FAIL_WORKFLOW,
            },
            EnumWorkflowState.COMPLETED: {},  # Terminal state
            EnumWorkflowState.FAILED: {
                EnumWorkflowState.PENDING: EnumTransitionEvent.RETRY_WORKFLOW,
            },
        }
    )

    # Guard conditions for transitions (state -> conditions)
    guard_conditions: dict[EnumWorkflowState, list[str]] = field(
        default_factory=lambda: {
            EnumWorkflowState.ANALYZING: ["has_requirements"],
            EnumWorkflowState.GENERATING: ["has_analysis_results"],
            EnumWorkflowState.VALIDATING: ["has_generated_code"],
            EnumWorkflowState.COMPLETED: ["validation_passed"],
        }
    )


# === FSM State Tracker ===


class FSMStateTracker:
    """
    FSM State Tracker for Code Generation Workflows.

    Manages workflow state transitions with:
    - State validation and guard conditions
    - Transition history tracking
    - Intent-based persistence (no direct I/O)
    - State recovery mechanisms
    - Performance metrics

    Pure function pattern:
    - No direct database operations
    - Emits intents for side effects
    - In-memory state management with periodic snapshots

    Performance:
    - <10ms per state transition
    - <5ms for state lookups
    - Support for 10,000+ concurrent workflows
    """

    def __init__(
        self,
        config: Optional[FSMConfiguration] = None,
        enable_postgres_persistence: bool = False,
    ) -> None:
        """
        Initialize FSM state tracker.

        Args:
            config: FSM configuration (defaults to standard codegen workflow)
            enable_postgres_persistence: Enable PostgreSQL persistence via intents
        """
        self.config = config or FSMConfiguration()
        self.enable_postgres_persistence = enable_postgres_persistence

        # Workflow state tracking
        self._workflow_states: dict[UUID, WorkflowStateRecord] = {}

        # Transition history (workflow_id -> [transitions])
        self._transition_history: dict[UUID, list[StateTransition]] = defaultdict(list)

        # Pending intents for batch emission
        self._pending_intents: list[ModelStateIntent] = []

        # Metrics
        self._total_transitions = 0
        self._failed_transitions = 0
        self._active_workflows = 0

        logger.info(
            f"FSMStateTracker initialized: "
            f"states={len(self.config.valid_states)}, "
            f"postgres_persistence={enable_postgres_persistence}"
        )

    async def initialize_workflow(
        self,
        workflow_id: UUID,
        initial_state: EnumWorkflowState = EnumWorkflowState.PENDING,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Initialize a new workflow in FSM.

        Args:
            workflow_id: Unique workflow identifier
            initial_state: Initial workflow state
            metadata: Optional workflow metadata

        Returns:
            True if initialization successful
        """
        if workflow_id in self._workflow_states:
            logger.warning(f"Workflow {workflow_id} already initialized")
            return False

        if initial_state not in self.config.valid_states:
            logger.error(f"Invalid initial state: {initial_state}")
            return False

        # Create workflow state record
        self._workflow_states[workflow_id] = WorkflowStateRecord(
            workflow_id=workflow_id,
            current_state=initial_state,
            previous_state=None,
            metadata=metadata or {},
        )

        self._active_workflows += 1

        # Emit persistence intent
        if self.enable_postgres_persistence:
            intent = ModelStateIntent(
                intent_type=EnumIntentType.PERSIST_STATE,
                workflow_id=workflow_id,
                state=initial_state,
                metadata={"action": "initialize", **(metadata or {})},
            )
            self._pending_intents.append(intent)

        logger.debug(f"Workflow {workflow_id} initialized in state {initial_state}")
        return True

    async def transition_state(
        self,
        workflow_id: UUID,
        to_state: EnumWorkflowState,
        event: EnumTransitionEvent,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Attempt to transition workflow to new state.

        Args:
            workflow_id: Workflow identifier
            to_state: Target state
            event: Transition event
            metadata: Optional transition metadata

        Returns:
            (success, error_message)
        """
        start_time = datetime.now(UTC)

        # Validate workflow exists
        if workflow_id not in self._workflow_states:
            return False, f"Workflow {workflow_id} not found"

        workflow = self._workflow_states[workflow_id]
        from_state = workflow.current_state

        # Check if transition is valid
        if to_state not in self.config.valid_transitions.get(from_state, {}):
            error = (
                f"Invalid transition: {from_state} -> {to_state} " f"(event: {event})"
            )
            self._failed_transitions += 1
            logger.warning(error)
            return False, error

        # Check if correct event for transition
        expected_event = self.config.valid_transitions[from_state][to_state]
        if event != expected_event:
            error = (
                f"Wrong event for transition {from_state} -> {to_state}: "
                f"got {event}, expected {expected_event}"
            )
            self._failed_transitions += 1
            logger.warning(error)
            return False, error

        # Check guard conditions
        guard_met, guard_reason = self._check_guard_conditions(to_state, metadata or {})
        if not guard_met:
            error = f"Guard condition failed for {to_state}: {guard_reason}"
            self._failed_transitions += 1
            logger.warning(error)
            return False, error

        # Perform transition
        workflow.previous_state = workflow.current_state
        workflow.current_state = to_state
        workflow.state_entry_time = datetime.now(UTC)
        workflow.transition_count += 1

        # Record transition
        transition = StateTransition(
            workflow_id=workflow_id,
            from_state=from_state,
            to_state=to_state,
            event=event,
            metadata=metadata or {},
            guard_conditions_met=True,
        )
        self._transition_history[workflow_id].append(transition)

        self._total_transitions += 1

        # Update active workflow count
        if to_state in self.config.terminal_states:
            self._active_workflows -= 1

        # Emit persistence intent
        if self.enable_postgres_persistence:
            intent = ModelStateIntent(
                intent_type=EnumIntentType.PERSIST_STATE,
                workflow_id=workflow_id,
                state=to_state,
                metadata={
                    "action": "transition",
                    "from_state": from_state.value,
                    "event": event.value,
                    **(metadata or {}),
                },
            )
            self._pending_intents.append(intent)

        # Performance tracking
        duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        if duration_ms > 10:
            logger.warning(f"State transition took {duration_ms:.2f}ms (target <10ms)")

        logger.debug(
            f"Workflow {workflow_id} transitioned: {from_state} -> {to_state} "
            f"(event: {event}, {duration_ms:.2f}ms)"
        )
        return True, None

    def _check_guard_conditions(
        self, state: EnumWorkflowState, metadata: dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Check guard conditions for state entry.

        Args:
            state: Target state
            metadata: Transition metadata

        Returns:
            (conditions_met, reason)
        """
        conditions = self.config.guard_conditions.get(state, [])
        if not conditions:
            return True, None

        # Check each condition
        for condition in conditions:
            if condition not in metadata or not metadata[condition]:
                return False, f"Missing required condition: {condition}"

        return True, None

    async def get_workflow_state(
        self, workflow_id: UUID
    ) -> Optional[WorkflowStateRecord]:
        """Get current workflow state."""
        return self._workflow_states.get(workflow_id)

    async def get_transition_history(self, workflow_id: UUID) -> list[StateTransition]:
        """Get transition history for workflow."""
        return self._transition_history.get(workflow_id, [])

    async def is_terminal_state(self, workflow_id: UUID) -> bool:
        """Check if workflow is in terminal state."""
        workflow = self._workflow_states.get(workflow_id)
        if not workflow:
            return False
        return workflow.current_state in self.config.terminal_states

    async def get_active_workflows_by_state(
        self, state: EnumWorkflowState
    ) -> list[UUID]:
        """Get all workflows in a specific state."""
        return [
            wf_id
            for wf_id, wf in self._workflow_states.items()
            if wf.current_state == state
        ]

    async def get_pending_intents(self) -> list[ModelStateIntent]:
        """Get and clear pending intents."""
        intents = self._pending_intents.copy()
        self._pending_intents.clear()
        return intents

    async def get_metrics(self) -> dict[str, Any]:
        """Get FSM tracker metrics."""
        state_distribution = defaultdict(int)
        for workflow in self._workflow_states.values():
            state_distribution[workflow.current_state.value] += 1

        return {
            "total_workflows": len(self._workflow_states),
            "active_workflows": self._active_workflows,
            "total_transitions": self._total_transitions,
            "failed_transitions": self._failed_transitions,
            "success_rate": (
                (self._total_transitions - self._failed_transitions)
                / self._total_transitions
                if self._total_transitions > 0
                else 0.0
            ),
            "state_distribution": dict(state_distribution),
            "pending_intents": len(self._pending_intents),
        }

    async def cleanup_terminal_workflows(self, older_than_hours: int = 24) -> int:
        """
        Clean up terminal workflows older than threshold.

        Args:
            older_than_hours: Remove terminal workflows older than this

        Returns:
            Number of workflows removed
        """
        now = datetime.now(UTC)
        cutoff = now.timestamp() - (older_than_hours * 3600)

        removed_count = 0
        workflows_to_remove = []

        for wf_id, workflow in self._workflow_states.items():
            # Only clean up terminal states
            if workflow.current_state not in self.config.terminal_states:
                continue

            # Check age
            if workflow.state_entry_time.timestamp() < cutoff:
                workflows_to_remove.append(wf_id)

        # Remove workflows
        for wf_id in workflows_to_remove:
            del self._workflow_states[wf_id]
            if wf_id in self._transition_history:
                del self._transition_history[wf_id]
            removed_count += 1

        if removed_count > 0:
            logger.info(
                f"Cleaned up {removed_count} terminal workflows "
                f"older than {older_than_hours}h"
            )

        return removed_count
