# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Batch Lifecycle Result Model for handler batch shutdown operations.

This module provides the ModelBatchLifecycleResult class for representing the
aggregate outcome of batch handler lifecycle operations (e.g., shutdown_handlers_by_priority).

Design Pattern:
    ModelBatchLifecycleResult replaces tuple[list[str], list[tuple[str, str | None]]]
    returns with a strongly-typed model that provides:
    - Clear separation of succeeded and failed results
    - Convenient helper methods for common queries
    - Full traceability via ModelLifecycleResult for each handler

Thread Safety:
    ModelBatchLifecycleResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.runtime.models import (
    ...     ModelBatchLifecycleResult,
    ...     ModelLifecycleResult,
    ... )
    >>>
    >>> # Create batch result from individual results
    >>> results = [
    ...     ModelLifecycleResult.succeeded("kafka"),
    ...     ModelLifecycleResult.succeeded("consul"),
    ...     ModelLifecycleResult.failed("db", "Timeout"),
    ... ]
    >>> batch = ModelBatchLifecycleResult.from_results(results)
    >>> batch.total_count
    3
    >>> batch.success_count
    2
    >>> batch.failure_count
    1
    >>> batch.all_succeeded
    False
"""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_lifecycle_result import ModelLifecycleResult


class ModelBatchLifecycleResult(BaseModel):
    """Result of a batch handler lifecycle operation (e.g., shutdown_handlers_by_priority).

    Provides a strongly-typed replacement for tuple[list[str], list[tuple[str, str | None]]]
    with clear separation of succeeded/failed handlers and helper methods.

    Attributes:
        results: List of all lifecycle results (both successes and failures).
        succeeded_handlers: List of handler types that completed successfully.
        failed_handlers: List of ModelLifecycleResult for handlers that failed.

    Example:
        >>> batch = ModelBatchLifecycleResult(
        ...     results=[
        ...         ModelLifecycleResult.succeeded("kafka"),
        ...         ModelLifecycleResult.failed("db", "Timeout"),
        ...     ],
        ...     succeeded_handlers=["kafka"],
        ...     failed_handlers=[ModelLifecycleResult.failed("db", "Timeout")],
        ... )
        >>> batch.has_failures
        True
        >>> batch.success_count
        1
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    results: list[ModelLifecycleResult] = Field(
        default_factory=list,
        description="All lifecycle results in execution order.",
    )
    succeeded_handlers: list[str] = Field(
        default_factory=list,
        description="Handler types that completed successfully.",
    )
    failed_handlers: list[ModelLifecycleResult] = Field(
        default_factory=list,
        description="Lifecycle results for handlers that failed.",
    )

    @classmethod
    def from_results(
        cls, results: list[ModelLifecycleResult]
    ) -> "ModelBatchLifecycleResult":
        """Create a batch result from a list of individual lifecycle results.

        Automatically separates succeeded and failed handlers.

        Args:
            results: List of individual ModelLifecycleResult instances.

        Returns:
            ModelBatchLifecycleResult with succeeded/failed handlers categorized.

        Example:
            >>> results = [
            ...     ModelLifecycleResult.succeeded("kafka"),
            ...     ModelLifecycleResult.failed("db", "Error"),
            ... ]
            >>> batch = ModelBatchLifecycleResult.from_results(results)
            >>> batch.succeeded_handlers
            ['kafka']
        """
        succeeded = [r.handler_type for r in results if r.success]
        failed = [r for r in results if not r.success]
        return cls(
            results=results,
            succeeded_handlers=succeeded,
            failed_handlers=failed,
        )

    @classmethod
    def empty(cls) -> "ModelBatchLifecycleResult":
        """Create an empty batch result (no handlers processed).

        Returns:
            Empty ModelBatchLifecycleResult.

        Example:
            >>> batch = ModelBatchLifecycleResult.empty()
            >>> batch.total_count
            0
        """
        return cls(results=[], succeeded_handlers=[], failed_handlers=[])

    @property
    def total_count(self) -> int:
        """Return total number of handlers processed.

        Returns:
            Count of all handlers (succeeded + failed).
        """
        return len(self.results)

    @property
    def success_count(self) -> int:
        """Return number of handlers that succeeded.

        Returns:
            Count of successfully completed handlers.
        """
        return len(self.succeeded_handlers)

    @property
    def failure_count(self) -> int:
        """Return number of handlers that failed.

        Returns:
            Count of failed handlers.
        """
        return len(self.failed_handlers)

    @property
    def all_succeeded(self) -> bool:
        """Check if all handlers completed successfully.

        Returns:
            True if no failures occurred, False otherwise.
        """
        return len(self.failed_handlers) == 0 and len(self.results) > 0

    @property
    def has_failures(self) -> bool:
        """Check if any handlers failed.

        Returns:
            True if at least one handler failed, False otherwise.
        """
        return len(self.failed_handlers) > 0

    def get_failure_messages(self) -> dict[str, str]:
        """Get a mapping of failed handler types to their error messages.

        Note:
            All failed handlers have non-empty error messages since
            ModelLifecycleResult.failed() requires a non-empty error_message.

        Returns:
            Dict mapping handler_type to error_message for all failures.

        Example:
            >>> batch = ModelBatchLifecycleResult.from_results([
            ...     ModelLifecycleResult.failed("db", "Timeout"),
            ...     ModelLifecycleResult.failed("cache", "Connection refused"),
            ... ])
            >>> batch.get_failure_messages()
            {'db': 'Timeout', 'cache': 'Connection refused'}
        """
        return {r.handler_type: r.error_message for r in self.failed_handlers}

    def __str__(self) -> str:
        """Return a human-readable string representation for debugging.

        Returns:
            String format showing succeeded and failed counts.
        """
        return (
            f"ModelBatchLifecycleResult("
            f"total={self.total_count}, "
            f"succeeded={self.success_count}, "
            f"failed={self.failure_count})"
        )


__all__ = ["ModelBatchLifecycleResult"]
