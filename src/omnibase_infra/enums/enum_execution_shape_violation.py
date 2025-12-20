# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Execution Shape Violation Enumeration.

Defines the canonical violation types for the ONEX execution shape validator.
These violations represent constraint breaches in the ONEX 4-node architecture
that would compromise system integrity or determinism guarantees.

Violation Categories:
    - Return Type Violations: Handler returning forbidden output types
    - Direct Publish Violations: Handler bypassing event bus routing
    - Determinism Violations: Handler accessing non-deterministic resources
    - Topic Mismatch Violations: Handler subscribed to wrong topic category

Note on PROJECTION Violations:
    PROJECTION is a valid node output type (EnumNodeOutputType.PROJECTION) that
    ONLY REDUCERs may produce. It is NOT a message routing category (not in
    EnumMessageCategory). This distinction is critical:

    - EnumNodeOutputType: Defines what a node CAN produce (includes PROJECTION)
    - EnumMessageCategory: Defines how messages are ROUTED (excludes PROJECTION)

    Projections represent aggregated state from REDUCER nodes. They are not
    routed via Kafka topics like EVENTs, COMMANDs, and INTENTs. Violations like
    ORCHESTRATOR_RETURNS_PROJECTIONS or EFFECT_RETURNS_PROJECTIONS indicate a
    non-REDUCER node is incorrectly trying to produce aggregated state output.

See Also:
    - EnumNodeOutputType: Defines valid node output types (includes PROJECTION)
    - EnumMessageCategory: Defines message categories for routing (excludes PROJECTION)
    - EnumHandlerType: Defines the 4-node architecture handler types
"""

from enum import Enum


class EnumExecutionShapeViolation(str, Enum):
    """Execution shape violation types for ONEX architecture compliance.

    These represent constraint violations detected during static analysis
    of ONEX handlers. Each violation type indicates a specific breach of
    the 4-node architecture rules.

    Return Type Violations:
        FORBIDDEN_RETURN_TYPE: Generic violation for handler returning forbidden type.
            Used as fallback when no specific violation type is mapped.
        REDUCER_RETURNS_EVENTS: Reducer handlers cannot emit events.
            Reducers manage state; event emission is an effect operation.
        ORCHESTRATOR_RETURNS_INTENTS: Orchestrators cannot emit intents.
            Intents originate from external systems, not orchestration.
        ORCHESTRATOR_RETURNS_PROJECTIONS: Orchestrators cannot produce projections.
            PROJECTION is a node output type (EnumNodeOutputType.PROJECTION), NOT a
            message routing category (EnumMessageCategory). Projections represent
            aggregated state and are ONLY valid for REDUCER node output types.
            Orchestrators coordinate workflows and should return COMMANDs or EVENTs.
        EFFECT_RETURNS_PROJECTIONS: Effect handlers cannot produce projections.
            PROJECTION is a node output type (EnumNodeOutputType.PROJECTION), NOT a
            message routing category (EnumMessageCategory). Projections represent
            aggregated state and are ONLY valid for REDUCER node output types.
            Effect handlers interact with external systems and should return
            EVENTs or COMMANDs.

    Operation Violations:
        HANDLER_DIRECT_PUBLISH: Handler bypasses event bus for direct publish.
            All message routing must go through the event bus abstraction.
        REDUCER_ACCESSES_SYSTEM_TIME: Reducer accesses non-deterministic time.
            Reducers must be deterministic for replay consistency.

    Topic Violations:
        TOPIC_CATEGORY_MISMATCH: Handler subscribed to wrong topic category.
            Event handlers should subscribe to event topics, etc.

    Routing Violations:
        UNMAPPED_MESSAGE_ROUTE: Message type is not registered in routing.
            All message types must have explicit route registrations.

    Syntax/Parse Violations:
        SYNTAX_ERROR: File contains invalid Python syntax.
            File could not be parsed for analysis due to syntax errors.
    """

    # Return type violations
    FORBIDDEN_RETURN_TYPE = "forbidden_return_type"
    REDUCER_RETURNS_EVENTS = "reducer_returns_events"
    ORCHESTRATOR_RETURNS_INTENTS = "orchestrator_returns_intents"
    ORCHESTRATOR_RETURNS_PROJECTIONS = "orchestrator_returns_projections"
    EFFECT_RETURNS_PROJECTIONS = "effect_returns_projections"

    # Operation violations
    HANDLER_DIRECT_PUBLISH = "handler_direct_publish"
    REDUCER_ACCESSES_SYSTEM_TIME = "reducer_accesses_system_time"

    # Topic violations
    TOPIC_CATEGORY_MISMATCH = "topic_category_mismatch"

    # Routing violations
    UNMAPPED_MESSAGE_ROUTE = "unmapped_message_route"

    # Syntax/parse violations
    SYNTAX_ERROR = "syntax_error"


__all__ = ["EnumExecutionShapeViolation"]
