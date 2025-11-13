"""Stub for omnibase_core.enums.enum_node_type"""

from enum import Enum


class EnumNodeType(str, Enum):
    """ONEX v2.0 Node Types."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"


__all__ = ["EnumNodeType"]
