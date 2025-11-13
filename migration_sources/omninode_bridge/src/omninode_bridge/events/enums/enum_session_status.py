"""Session status enumeration for code generation workflows.

This enum defines the possible states of a code generation session
throughout its lifecycle.
"""

from enum import Enum


class EnumSessionStatus(str, Enum):
    """
    Status values for code generation sessions.

    Inherits from str for JSON serialization compatibility with Pydantic v2.

    Values:
        PENDING: Session created but not yet started
        PROCESSING: Session actively processing (analyzing, generating, validating)
        COMPLETED: Session completed successfully
        FAILED: Session failed with errors
        CANCELLED: Session cancelled by user or system
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
