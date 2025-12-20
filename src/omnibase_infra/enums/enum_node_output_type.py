# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Output Type Enumeration for ONEX Execution Shape Validation.

Defines the valid output types that ONEX nodes can produce. This enum is used
for execution shape validation to ensure nodes produce only allowed output types
based on their handler type (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).

IMPORTANT: This enum is distinct from EnumMessageCategory:
    - EnumMessageCategory: Defines message categories for routing (Kafka topics)
    - EnumNodeOutputType: Defines valid output types for execution shape validation

Key Difference - PROJECTION:
    PROJECTION is a valid node output type (REDUCERs can produce projections)
    but is NOT a message routing category (projections are not routed via Kafka
    topics in the same way as EVENTs, COMMANDs, and INTENTs).

Output Type Constraints by Handler Type:
    - EFFECT: Can output EVENT, COMMAND (external interaction results)
    - COMPUTE: Can output EVENT, COMMAND, INTENT (pure transformations)
    - REDUCER: Can output PROJECTION only (state consolidation)
    - ORCHESTRATOR: Can output COMMAND, EVENT (workflow coordination)

See Also:
    - EnumHandlerType: Defines the 4-node architecture handler types
    - EnumMessageCategory: Defines message categories for topic routing
    - EnumExecutionShapeViolation: Defines validation violation types
"""

from enum import Enum


class EnumNodeOutputType(str, Enum):
    """Valid output types for ONEX 4-node architecture execution shape validation.

    This enum defines what types of outputs a node can produce. The execution
    shape validator uses this to ensure nodes only produce outputs allowed
    for their handler type.

    This is NOT the same as EnumMessageCategory which defines how messages
    are routed through Kafka topics. EnumNodeOutputType is specifically for
    validating node execution contracts.

    Attributes:
        EVENT: Domain events representing facts about what happened.
            Produced by: EFFECT, COMPUTE, ORCHESTRATOR
            Example outputs: OrderCreatedEvent, PaymentProcessedEvent
        COMMAND: Commands requesting an action to be performed.
            Produced by: EFFECT, COMPUTE, ORCHESTRATOR
            Example outputs: ProcessPaymentCommand, SendNotificationCommand
        INTENT: User intents requiring validation before processing.
            Produced by: COMPUTE only (transforms user input to validated intent)
            Example outputs: ValidatedCheckoutIntent, ApprovedTransferIntent
        PROJECTION: State projections for read model optimization.
            Produced by: REDUCER only (state consolidation output)
            Example outputs: OrderSummaryProjection, UserProfileProjection
            NOTE: PROJECTION is valid here but NOT in EnumMessageCategory
            because projections are node outputs, not routed messages.

    Example:
        >>> from omnibase_infra.enums import EnumNodeOutputType, EnumHandlerType
        >>>
        >>> # Validate that a REDUCER node can produce PROJECTION
        >>> handler_type = EnumHandlerType.REDUCER
        >>> output_type = EnumNodeOutputType.PROJECTION
        >>> # PROJECTION is valid for REDUCER
        >>>
        >>> # Validate that an EFFECT node cannot produce PROJECTION
        >>> handler_type = EnumHandlerType.EFFECT
        >>> output_type = EnumNodeOutputType.PROJECTION
        >>> # This would be an execution shape violation
    """

    EVENT = "event"
    COMMAND = "command"
    INTENT = "intent"
    PROJECTION = "projection"

    def is_event(self) -> bool:
        """Check if this is an EVENT output type."""
        return self == EnumNodeOutputType.EVENT

    def is_command(self) -> bool:
        """Check if this is a COMMAND output type."""
        return self == EnumNodeOutputType.COMMAND

    def is_intent(self) -> bool:
        """Check if this is an INTENT output type."""
        return self == EnumNodeOutputType.INTENT

    def is_projection(self) -> bool:
        """Check if this is a PROJECTION output type."""
        return self == EnumNodeOutputType.PROJECTION


__all__ = ["EnumNodeOutputType"]
