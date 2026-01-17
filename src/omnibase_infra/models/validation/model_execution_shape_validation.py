# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Execution Shape Validation Model.

Validates that message categories (EVENT, COMMAND, INTENT) can be processed
by specific node kinds (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR) according to
ONEX architectural patterns.

Design Pattern:
    ModelExecutionShapeValidation uses static rules to validate that a
    message category can be routed to a target node kind. This ensures
    ONEX architectural compliance at registration time.

Thread Safety:
    All methods are pure functions operating on immutable data,
    making this class fully thread-safe.

Valid Execution Shapes (source_category -> target_node_kinds):
    - EVENT -> REDUCER, COMPUTE (events are consumed by reducers/projections)
    - COMMAND -> EFFECT, ORCHESTRATOR (commands trigger actions/workflows)
    - INTENT -> ORCHESTRATOR (intents require interpretation/routing)

Invalid Shapes:
    - EVENT -> EFFECT (events should not trigger external I/O directly)
    - EVENT -> ORCHESTRATOR (events are facts, not routing decisions)
    - COMMAND -> REDUCER (commands should trigger actions, not aggregation)
    - COMMAND -> COMPUTE (commands should trigger actions, not transforms)
    - INTENT -> EFFECT (intents need interpretation first)
    - INTENT -> REDUCER (intents need interpretation first)
    - INTENT -> COMPUTE (intents need interpretation first)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.enums import EnumMessageCategory

# Valid execution shapes: category -> allowed node kinds
_VALID_SHAPES: dict[EnumMessageCategory, set[EnumNodeKind]] = {
    EnumMessageCategory.EVENT: {EnumNodeKind.REDUCER, EnumNodeKind.COMPUTE},
    EnumMessageCategory.COMMAND: {EnumNodeKind.EFFECT, EnumNodeKind.ORCHESTRATOR},
    EnumMessageCategory.INTENT: {EnumNodeKind.ORCHESTRATOR},
}

# Rationales for invalid shapes
_INVALID_RATIONALES: dict[tuple[EnumMessageCategory, EnumNodeKind], str] = {
    (EnumMessageCategory.EVENT, EnumNodeKind.EFFECT): (
        "Events should not trigger external I/O directly. "
        "Use a reducer or compute node to process events first."
    ),
    (EnumMessageCategory.EVENT, EnumNodeKind.ORCHESTRATOR): (
        "Events are immutable facts, not routing decisions. "
        "Use a reducer to aggregate events."
    ),
    (EnumMessageCategory.COMMAND, EnumNodeKind.REDUCER): (
        "Commands should trigger actions, not aggregation. "
        "Use an effect node to execute commands."
    ),
    (EnumMessageCategory.COMMAND, EnumNodeKind.COMPUTE): (
        "Commands should trigger actions, not pure transforms. "
        "Use an effect or orchestrator node."
    ),
    (EnumMessageCategory.INTENT, EnumNodeKind.EFFECT): (
        "Intents need interpretation before triggering I/O. "
        "Use an orchestrator to route intents."
    ),
    (EnumMessageCategory.INTENT, EnumNodeKind.REDUCER): (
        "Intents need interpretation first. "
        "Use an orchestrator to interpret and route intents."
    ),
    (EnumMessageCategory.INTENT, EnumNodeKind.COMPUTE): (
        "Intents need interpretation first. "
        "Use an orchestrator to interpret and route intents."
    ),
}


class ModelExecutionShapeValidation(BaseModel):
    """
    Result of validating an execution shape.

    Contains the validation result and rationale for why a particular
    message category -> node kind combination is valid or invalid.

    Attributes:
        source_category: The message category being routed.
        target_node_kind: The target node kind for processing.
        is_allowed: Whether this execution shape is valid.
        rationale: Explanation of why the shape is valid/invalid.

    Example:
        >>> validation = ModelExecutionShapeValidation.validate_shape(
        ...     source_category=EnumMessageCategory.EVENT,
        ...     target_node_kind=EnumNodeKind.REDUCER,
        ... )
        >>> validation.is_allowed
        True
        >>> validation.rationale
        'EVENT -> REDUCER is a valid execution shape for event aggregation.'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    source_category: EnumMessageCategory = Field(
        ...,
        description="The message category being routed.",
    )
    target_node_kind: EnumNodeKind = Field(
        ...,
        description="The target node kind for processing.",
    )
    is_allowed: bool = Field(
        ...,
        description="Whether this execution shape is valid.",
    )
    rationale: str = Field(
        ...,
        description="Explanation of why the shape is valid/invalid.",
    )

    @classmethod
    def validate_shape(
        cls,
        source_category: EnumMessageCategory,
        target_node_kind: EnumNodeKind,
    ) -> ModelExecutionShapeValidation:
        """
        Validate that a message category can be routed to a node kind.

        Args:
            source_category: The message category (EVENT, COMMAND, INTENT).
            target_node_kind: The target node kind (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).

        Returns:
            ModelExecutionShapeValidation with validation result.

        Example:
            >>> # Valid shape: EVENT -> REDUCER
            >>> validation = ModelExecutionShapeValidation.validate_shape(
            ...     source_category=EnumMessageCategory.EVENT,
            ...     target_node_kind=EnumNodeKind.REDUCER,
            ... )
            >>> validation.is_allowed
            True

            >>> # Invalid shape: EVENT -> EFFECT
            >>> validation = ModelExecutionShapeValidation.validate_shape(
            ...     source_category=EnumMessageCategory.EVENT,
            ...     target_node_kind=EnumNodeKind.EFFECT,
            ... )
            >>> validation.is_allowed
            False
        """
        allowed_kinds = _VALID_SHAPES.get(source_category, set())
        is_allowed = target_node_kind in allowed_kinds

        if is_allowed:
            rationale = (
                f"{source_category.value.upper()} -> {target_node_kind.value.upper()} "
                f"is a valid execution shape."
            )
        else:
            # Get specific rationale or use generic message
            key = (source_category, target_node_kind)
            rationale = _INVALID_RATIONALES.get(
                key,
                f"{source_category.value.upper()} -> {target_node_kind.value.upper()} "
                f"is not a valid execution shape.",
            )

        return cls(
            source_category=source_category,
            target_node_kind=target_node_kind,
            is_allowed=is_allowed,
            rationale=rationale,
        )

    @classmethod
    def get_valid_node_kinds(cls, category: EnumMessageCategory) -> set[EnumNodeKind]:
        """
        Get the valid target node kinds for a message category.

        Args:
            category: The message category to check.

        Returns:
            Set of valid node kinds for this category.

        Example:
            >>> kinds = ModelExecutionShapeValidation.get_valid_node_kinds(
            ...     EnumMessageCategory.EVENT
            ... )
            >>> EnumNodeKind.REDUCER in kinds
            True
            >>> EnumNodeKind.EFFECT in kinds
            False
        """
        return _VALID_SHAPES.get(category, set()).copy()


__all__ = ["ModelExecutionShapeValidation"]
