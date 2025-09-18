"""
Workflow step type enum.

Strongly typed values for workflow step types.
"""

from enum import Enum


class EnumWorkflowStepType(str, Enum):
    """Strongly typed workflow step type values."""

    AGENT_EXECUTION = "agent_execution"
    DATA_PROCESSING = "data_processing"
    VALIDATION = "validation"
    COORDINATION = "coordination"
    CLEANUP = "cleanup"


# Export for use
__all__ = ["EnumWorkflowStepType"]