"""
Workflow step category enum.

Strongly typed values for workflow step categories.
"""

from enum import Enum


class EnumWorkflowStepCategory(str, Enum):
    """Strongly typed workflow step category values."""

    INITIALIZATION = "initialization"
    PROCESSING = "processing"
    VALIDATION = "validation"
    FINALIZATION = "finalization"
    ERROR_HANDLING = "error_handling"


# Export for use
__all__ = ["EnumWorkflowStepCategory"]