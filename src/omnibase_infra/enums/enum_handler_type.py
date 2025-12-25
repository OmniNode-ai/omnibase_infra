# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Type Enumeration for ONEX 4-Node Architecture.

Defines the canonical handler types for the ONEX execution shape validator.
Each handler type corresponds to a node category in the ONEX 4-node architecture
and has specific constraints on allowed output types and operations.

Handler Type Output Constraints (see EnumNodeOutputType for output types):
    - EFFECT: External interactions (API calls, DB queries, file I/O).
              Can output EVENT, COMMAND but not PROJECTION.
    - COMPUTE: Pure data processing and business logic transformations.
               Can output EVENT, COMMAND, INTENT. No side effects.
    - REDUCER: State management and persistence operations.
               TARGET DESIGN (post-OMN-914): Can output PROJECTION only.
               Cannot output EVENT. Cannot access system time (must be deterministic).
               MVP INTERIM: Current reducers (e.g., RegistrationReducer) may emit
               intents that become EVENTs/COMMANDs. This will be enforced once
               full purity gates are in place per OMN-914.
    - ORCHESTRATOR: Workflow coordination and step sequencing.
                    Can output COMMAND, EVENT but not INTENT or PROJECTION.

Note on PROJECTION:
    PROJECTION is a node output type (EnumNodeOutputType.PROJECTION), not a
    message routing category (not in EnumMessageCategory). Only REDUCERs may
    produce PROJECTION outputs for state consolidation.

See Also:
    - EnumNodeOutputType: Defines valid node output types (includes PROJECTION)
    - EnumMessageCategory: Defines message categories for routing (excludes PROJECTION)
    - EnumExecutionShapeViolation: Defines validation violation types
"""

from enum import Enum


class EnumHandlerType(str, Enum):
    """Handler types for ONEX 4-node architecture execution shape validation.

    These represent the four canonical node types in the ONEX architecture.
    Each type has specific constraints on what operations are allowed and
    what output types can be produced (see EnumNodeOutputType).

    Attributes:
        EFFECT: External interaction handlers (API calls, DB operations).
            Responsible for side effects and external system integration.
            Can output: EVENT, COMMAND. Cannot output: PROJECTION.
        COMPUTE: Pure data processing and transformation handlers.
            No side effects, deterministic transformations.
            Can output: EVENT, COMMAND, INTENT.
        REDUCER: State management and persistence handlers.
            Manages state consolidation and projections.
            TARGET DESIGN (post-OMN-914): Can output PROJECTION only.
            Cannot output: EVENT. Cannot access system time (must be deterministic).
            MVP INTERIM: Current reducers may emit intents that become EVENTs/COMMANDs
            until full purity gates are enforced per OMN-914.
        ORCHESTRATOR: Workflow coordination handlers.
            Coordinates multi-step workflows and routing.
            Can output: COMMAND, EVENT. Cannot output: INTENT, PROJECTION.

    Note:
        Output types refer to EnumNodeOutputType values. PROJECTION is a node
        output type, not a message routing category (not in EnumMessageCategory).
    """

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"


__all__ = ["EnumHandlerType"]
