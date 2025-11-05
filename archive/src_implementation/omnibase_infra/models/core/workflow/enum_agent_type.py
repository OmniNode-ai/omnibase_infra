"""
Agent type enum.

Strongly typed values for agent types in workflow execution.
"""

from enum import Enum


class EnumAgentType(str, Enum):
    """Strongly typed agent type values."""

    COORDINATOR = "coordinator"
    PROCESSOR = "processor"
    VALIDATOR = "validator"
    SPECIALIST = "specialist"


# Export for use
__all__ = ["EnumAgentType"]
