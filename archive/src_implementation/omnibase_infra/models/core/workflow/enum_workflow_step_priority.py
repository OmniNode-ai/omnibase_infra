"""
Workflow step priority enum.

Strongly typed values for workflow step execution priority.
"""

from enum import Enum


class EnumWorkflowStepPriority(str, Enum):
    """Strongly typed workflow step priority values."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# Export for use
__all__ = ["EnumWorkflowStepPriority"]
