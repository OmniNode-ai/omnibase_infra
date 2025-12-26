# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Operation Context Model for handler tracking.

This module provides a Pydantic model for encapsulating operation context
during handler execution, enabling consistent tracking and tracing.
"""

from __future__ import annotations

import time
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelOperationContext(BaseModel):
    """Encapsulates operation context for handler tracking.

    This model provides a strongly-typed container for operation metadata,
    replacing scattered local variables with a cohesive context object.

    Attributes:
        correlation_id: UUID for distributed tracing across services
        operation_name: Name of the operation being executed
        started_at: Unix timestamp when the operation started
        timeout_seconds: Maximum allowed execution time
        metadata: Additional key-value pairs for operation-specific context

    Example:
        >>> ctx = ModelOperationContext.create("consul.kv_get")
        >>> ctx.operation_name
        'consul.kv_get'
        >>> ctx.is_timed_out()
        False

    ID Semantics:
        correlation_id: Groups related operations across distributed services.
            Used for filtering logs, tracing request flows, and debugging.
            Propagated from request envelopes for consistent distributed tracing.

    Security:
        The metadata field should contain SANITIZED values only.
        Never include credentials, tokens, or other sensitive data.
        See CLAUDE.md "Error Sanitization Guidelines" for the security policy.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    correlation_id: UUID = Field(
        description="UUID for distributed tracing across services",
    )
    operation_name: str = Field(
        description="Name of the operation being executed",
    )
    started_at: float = Field(
        description="Unix timestamp when the operation started",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.0,
        le=3600.0,
        description="Maximum allowed execution time in seconds",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value pairs for operation-specific context",
    )

    @classmethod
    def create(
        cls,
        operation_name: str,
        correlation_id: UUID | None = None,
        timeout_seconds: float = 30.0,
        metadata: dict[str, str] | None = None,
    ) -> ModelOperationContext:
        """Create a new operation context with current timestamp.

        Args:
            operation_name: Name of the operation being executed
            correlation_id: Optional UUID for tracing (auto-generated if None)
            timeout_seconds: Maximum allowed execution time (default: 30.0)
            metadata: Optional additional key-value pairs

        Returns:
            New ModelOperationContext with current timestamp.

        Example:
            >>> ctx = ModelOperationContext.create("vault.read_secret")
            >>> ctx.operation_name
            'vault.read_secret'
        """
        return cls(
            correlation_id=correlation_id if correlation_id is not None else uuid4(),
            operation_name=operation_name,
            started_at=time.time(),
            timeout_seconds=timeout_seconds,
            metadata=metadata if metadata is not None else {},
        )

    def elapsed_seconds(self) -> float:
        """Calculate elapsed time since operation started.

        Returns:
            Elapsed time in seconds since started_at.

        Note:
            Calls time.time() on each invocation. Multiple calls within
            a tight loop may return slightly different values.
        """
        return time.time() - self.started_at

    def remaining_seconds(self) -> float:
        """Calculate remaining time before timeout.

        Returns:
            Remaining time in seconds (can be negative if timed out).

        Note:
            Calls elapsed_seconds() internally, which uses time.time().
            Multiple calls may return slightly different values.
        """
        return self.timeout_seconds - self.elapsed_seconds()

    def is_timed_out(self) -> bool:
        """Check if the operation has exceeded its timeout.

        Returns:
            True if elapsed time exceeds timeout_seconds, False otherwise.

        Note:
            Calls elapsed_seconds() internally, which uses time.time().
            Result may change between calls if near the timeout boundary.
        """
        return self.elapsed_seconds() > self.timeout_seconds

    def with_metadata(self, key: str, value: str) -> ModelOperationContext:
        """Create a copy with additional metadata.

        Args:
            key: Metadata key
            value: Metadata value (must be sanitized)

        Returns:
            New ModelOperationContext with updated metadata.

        Note:
            Values should be sanitized before calling this method.
        """
        new_metadata = dict(self.metadata)
        new_metadata[key] = value
        return ModelOperationContext(
            correlation_id=self.correlation_id,
            operation_name=self.operation_name,
            started_at=self.started_at,
            timeout_seconds=self.timeout_seconds,
            metadata=new_metadata,
        )

    def with_operation_name(self, operation_name: str) -> ModelOperationContext:
        """Create a copy with a different operation name.

        Useful when a handler routes to sub-operations while preserving
        the correlation_id and timing context.

        Args:
            operation_name: New operation name

        Returns:
            New ModelOperationContext with updated operation_name.
        """
        return ModelOperationContext(
            correlation_id=self.correlation_id,
            operation_name=operation_name,
            started_at=self.started_at,
            timeout_seconds=self.timeout_seconds,
            metadata=self.metadata,
        )


__all__: list[str] = ["ModelOperationContext"]
