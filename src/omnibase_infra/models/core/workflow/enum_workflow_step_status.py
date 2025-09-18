"""
Workflow step status enum.

Strongly typed status values for workflow step execution.
"""

from enum import Enum


class EnumWorkflowStepStatus(str, Enum):
    """Strongly typed workflow step status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


# Export for use
__all__ = ["EnumWorkflowStepStatus"]