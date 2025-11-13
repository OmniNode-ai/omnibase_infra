"""Node type enumeration for ONEX architecture.

This enum defines the four primary node types in the ONEX architecture,
used for node classification and validation.
"""

from enum import Enum


class EnumNodeType(str, Enum):
    """
    ONEX node type classification.

    Inherits from str for JSON serialization compatibility with Pydantic v2.

    Values:
        ORCHESTRATOR: Workflow coordination and multi-step execution
        COMPUTE: Pure transformations and algorithmic processing
        REDUCER: Aggregation, persistence, and state management
        EFFECT: External I/O, APIs, and side effects
    """

    ORCHESTRATOR = "orchestrator"
    COMPUTE = "compute"
    REDUCER = "reducer"
    EFFECT = "effect"
