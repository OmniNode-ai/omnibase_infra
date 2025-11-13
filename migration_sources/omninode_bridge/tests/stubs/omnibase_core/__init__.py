"""
Stub module for omnibase_core - Minimal implementation for testing.

This stub provides the essential classes required by omninode_bridge
without pulling in the full ONEX framework dependency.

For production deployments, use the full omnibase_core package from
the Archon repository.
"""

from enum import Enum
from typing import Any, Optional


class EnumCoreErrorCode(str, Enum):
    """
    Core error codes for ONEX-compliant error handling.

    This is a minimal stub implementation containing only the error codes
    currently used by omninode_bridge components.
    """

    # Validation errors
    VALIDATION_ERROR = "validation_error"
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
    INVALID_PARAMETER = "invalid_parameter"
    INVALID_INPUT = "invalid_input"

    # Database errors
    DATABASE_OPERATION_ERROR = "database_operation_error"
    DATABASE_CONNECTION_ERROR = "database_connection_error"

    # Infrastructure errors
    DEPENDENCY_ERROR = "dependency_error"
    INTERNAL_ERROR = "internal_error"
    EXECUTION_ERROR = "execution_error"
    NOT_IMPLEMENTED = "not_implemented"
    SERVICE_UNAVAILABLE = "service_unavailable"

    # External service errors
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    TIMEOUT = "timeout"

    # Kafka/messaging errors
    KAFKA_PUBLISH_ERROR = "kafka_publish_error"
    KAFKA_CONSUME_ERROR = "kafka_consume_error"


class ModelOnexError(Exception):
    """
    ONEX-compliant error model with structured context.

    Lightweight stub implementation that inherits from Exception only
    to avoid metaclass conflicts. Uses regular attributes instead of
    Pydantic fields for testing purposes.

    Attributes:
        code: Error classification from EnumCoreErrorCode
        message: Human-readable error description
        context: Additional structured context (optional)
    """

    def __init__(
        self,
        code: Optional[EnumCoreErrorCode] = None,
        message: str = "",
        context: Optional[dict[str, Any]] = None,
        error_code: Optional[EnumCoreErrorCode] = None,
        details: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize ONEX error with code, message, and optional context.

        Args:
            code: Error classification from EnumCoreErrorCode (legacy)
            error_code: Error classification from EnumCoreErrorCode (preferred)
            message: Human-readable error description
            context: Additional structured context (optional, legacy)
            details: Additional structured context (optional, preferred)
        """
        super().__init__(message)
        # Support both old and new parameter names
        self.code = error_code or code
        self.error_code = error_code or code
        self.message = message
        self.context = details or context or {}
        self.details = details or context or {}

    def __str__(self) -> str:
        """String representation for logging."""
        if self.context:
            return f"[{self.code.value}] {self.message} | Context: {self.context}"
        return f"[{self.code.value}] {self.message}"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"ModelOnexError(code={self.code.value}, message={self.message!r}, context={self.context})"


# Public API
__all__ = ["EnumCoreErrorCode", "ModelOnexError"]
