# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Type Enumeration for ONEX 4-Node Architecture.

Defines the canonical handler types for the ONEX execution shape validator.
Each handler type corresponds to a node category in the ONEX 4-node architecture
and has specific constraints on allowed message types and operations.

Handler Type Constraints:
    - EFFECT: External interactions (API calls, DB queries, file I/O).
              Can return EVENTs and COMMANDs but not PROJECTIONs.
    - COMPUTE: Pure data processing and business logic transformations.
               Can return any message type, no side effects.
    - REDUCER: State management and persistence operations.
               Can return PROJECTIONs but not EVENTs.
               Cannot access system time (must be deterministic).
    - ORCHESTRATOR: Workflow coordination and step sequencing.
                    Can return COMMANDs and EVENTs but not INTENTs or PROJECTIONs.
"""

from enum import Enum


class EnumHandlerType(str, Enum):
    """Handler types for ONEX 4-node architecture execution shape validation.

    These represent the four canonical node types in the ONEX architecture.
    Each type has specific constraints on what operations are allowed and
    what message types can be produced.

    Attributes:
        EFFECT: External interaction handlers (API calls, DB operations).
            Responsible for side effects and external system integration.
            Cannot return PROJECTION messages.
        COMPUTE: Pure data processing and transformation handlers.
            No side effects, deterministic transformations.
            Can produce any message type.
        REDUCER: State management and persistence handlers.
            Manages state consolidation and projections.
            Cannot return EVENT messages or access system time.
        ORCHESTRATOR: Workflow coordination handlers.
            Coordinates multi-step workflows and routing.
            Cannot return INTENT or PROJECTION messages.
    """

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"


__all__ = ["EnumHandlerType"]
