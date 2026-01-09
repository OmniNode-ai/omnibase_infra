# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Archetype Enumeration for ONEX 4-Node Architecture.

Defines the canonical node archetypes for the ONEX execution shape validator.
Each archetype corresponds to a node category in the ONEX 4-node architecture
and has specific constraints on allowed output types and operations.

Node Archetype Output Constraints (see EnumNodeOutputType for output types):
    - EFFECT: External interactions (API calls, DB queries, file I/O).
              Can output EVENT, COMMAND but not PROJECTION.
    - COMPUTE: Pure data processing and business logic transformations.
               Can output EVENT, COMMAND, INTENT. No side effects.
    - REDUCER: State management and persistence operations.
               Can output PROJECTION only. Cannot output EVENT.
               Must be deterministic (no system time access).
    - ORCHESTRATOR: Workflow coordination and step sequencing.
                    Can output COMMAND, EVENT but not INTENT or PROJECTION.

Note on Design:
    This enum is distinct from:
    - EnumHandlerType: Architectural role (how handler binds to system)
    - EnumHandlerTypeCategory: Behavioral classification (side effects)

    EnumNodeArchetype specifically captures the 4-node architecture for
    execution shape validation which defines what outputs each node can produce.

See Also:
    - EnumNodeOutputType: Defines valid node output types (includes PROJECTION)
    - EnumMessageCategory: Defines message categories for routing (excludes PROJECTION)
    - EnumExecutionShapeViolation: Defines validation violation types
    - EnumHandlerType: Handler architectural role (INFRA_HANDLER, NODE_HANDLER, etc.)
    - EnumHandlerTypeCategory: Handler behavioral classification (EFFECT, COMPUTE, etc.)
    - NODE_ARCHETYPE_EXPECTED_CATEGORIES: Maps archetypes to expected input/output types
      (see topic_category_validator.py for the mapping and its dual-purpose documentation)
"""

from enum import Enum


class EnumNodeArchetype(str, Enum):
    """Node archetypes for ONEX 4-node architecture execution shape validation.

    These represent the four canonical node types in the ONEX architecture.
    Each type has specific constraints on what operations are allowed and
    what output types can be produced (see EnumNodeOutputType).

    Attributes:
        EFFECT: External interaction nodes (API calls, DB operations).
            Responsible for side effects and external system integration.
            Can output: EVENT, COMMAND. Cannot output: PROJECTION.
        COMPUTE: Pure data processing and transformation nodes.
            No side effects, deterministic transformations.
            Can output: EVENT, COMMAND, INTENT.
        REDUCER: State management and persistence nodes.
            Manages state consolidation and projections.
            Can output: PROJECTION only. Cannot output: EVENT.
            Must be deterministic (no system time access).
        ORCHESTRATOR: Workflow coordination nodes.
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


__all__ = ["EnumNodeArchetype"]
