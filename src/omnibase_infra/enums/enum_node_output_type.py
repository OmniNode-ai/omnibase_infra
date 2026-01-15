# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Output Type Enumeration for ONEX Execution Shape Validation.

Defines the valid output types that ONEX nodes can produce. This enum is used
for execution shape validation to ensure nodes produce only allowed output types
based on their node archetype (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).

IMPORTANT: This enum is distinct from EnumMessageCategory:
    - EnumMessageCategory: Defines message categories for routing (Kafka topics)
    - EnumNodeOutputType: Defines valid output types for execution shape validation

Key Difference - PROJECTION:
    PROJECTION is a valid node output type (REDUCERs can produce projections)
    but is NOT a message routing category (projections are not routed via Kafka
    topics in the same way as EVENTs, COMMANDs, and INTENTs).

Output Type Constraints by Node Archetype:
    - EFFECT: Can output EVENT, COMMAND (external interaction results)
    - COMPUTE: Can output EVENT, COMMAND, INTENT (pure transformations)
    - REDUCER: Can output PROJECTION only (state consolidation)
    - ORCHESTRATOR: Can output COMMAND, EVENT (workflow coordination)

See Also:
    - EnumNodeArchetype: Defines the 4-node architecture node archetypes
    - EnumMessageCategory: Defines message categories for topic routing
    - EnumExecutionShapeViolation: Defines validation violation types
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from omnibase_infra.enums.enum_message_category import EnumMessageCategory


class EnumNodeOutputType(str, Enum):
    """Valid output types for ONEX 4-node architecture execution shape validation.

    This enum defines what types of outputs a node can produce. The execution
    shape validator uses this to ensure nodes only produce outputs allowed
    for their node archetype.

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
        >>> from omnibase_infra.enums import EnumNodeOutputType, EnumNodeArchetype
        >>>
        >>> # Validate that a REDUCER node can produce PROJECTION
        >>> node_archetype = EnumNodeArchetype.REDUCER
        >>> output_type = EnumNodeOutputType.PROJECTION
        >>> # PROJECTION is valid for REDUCER
        >>>
        >>> # Validate that an EFFECT node cannot produce PROJECTION
        >>> node_archetype = EnumNodeArchetype.EFFECT
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

    def is_routable(self) -> bool:
        """Check if this output type can be routed as a message.

        EVENT, COMMAND, and INTENT are routable message categories that can
        be published to Kafka topics. PROJECTION is NOT routable - it is a
        node output type used for state consolidation in REDUCER nodes.

        Returns:
            True if this output type can be routed as a message, False otherwise.

        Example:
            >>> EnumNodeOutputType.EVENT.is_routable()
            True
            >>> EnumNodeOutputType.COMMAND.is_routable()
            True
            >>> EnumNodeOutputType.INTENT.is_routable()
            True
            >>> EnumNodeOutputType.PROJECTION.is_routable()
            False
        """
        return self != EnumNodeOutputType.PROJECTION

    def to_message_category(self) -> EnumMessageCategory:
        """Convert this output type to the corresponding message category.

        Maps routable output types to their message category equivalents:
        - EVENT -> EnumMessageCategory.EVENT
        - COMMAND -> EnumMessageCategory.COMMAND
        - INTENT -> EnumMessageCategory.INTENT

        PROJECTION cannot be converted because it is not a routable message
        category - it is a node output type specific to REDUCER nodes.

        Returns:
            The corresponding EnumMessageCategory for this output type.

        Raises:
            ProtocolConfigurationError: If this is PROJECTION, which cannot be
                converted to a message category.

        Example:
            >>> from omnibase_infra.enums import EnumNodeOutputType
            >>> EnumNodeOutputType.EVENT.to_message_category()
            <EnumMessageCategory.EVENT: 'event'>
            >>> EnumNodeOutputType.COMMAND.to_message_category()
            <EnumMessageCategory.COMMAND: 'command'>
            >>> EnumNodeOutputType.INTENT.to_message_category()
            <EnumMessageCategory.INTENT: 'intent'>
            >>> EnumNodeOutputType.PROJECTION.to_message_category()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            ProtocolConfigurationError: Invalid output type for message category: expected EVENT, COMMAND, or INTENT, got 'projection'
        """
        # Import at runtime to avoid circular imports
        from omnibase_infra.enums.enum_infra_transport_type import (
            EnumInfraTransportType,
        )
        from omnibase_infra.enums.enum_message_category import EnumMessageCategory
        from omnibase_infra.errors import (
            ModelInfraErrorContext,
            ProtocolConfigurationError,
        )

        if self == EnumNodeOutputType.EVENT:
            return EnumMessageCategory.EVENT
        if self == EnumNodeOutputType.COMMAND:
            return EnumMessageCategory.COMMAND
        if self == EnumNodeOutputType.INTENT:
            return EnumMessageCategory.INTENT

        # PROJECTION cannot be converted to message category
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="to_message_category",
            correlation_id=uuid4(),
        )
        raise ProtocolConfigurationError(
            f"Invalid output type for message category: expected EVENT, COMMAND, or INTENT, got '{self.value}'",
            context=context,
            output_type=self.value,
        )


__all__ = ["EnumNodeOutputType"]
