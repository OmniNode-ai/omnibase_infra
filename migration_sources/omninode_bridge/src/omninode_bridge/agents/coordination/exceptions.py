"""
Custom exceptions for ThreadSafeState and agent coordination.

This module provides a hierarchy of typed exceptions for state management
operations with structured context for debugging and error handling.
"""

from typing import Optional


class ThreadSafeStateError(Exception):
    """Base exception for ThreadSafeState errors."""

    pass


class StateKeyError(ThreadSafeStateError, KeyError):
    """
    Raised when a required key is not found in state.

    Attributes:
        key: The key that was not found
        available_keys: List of keys currently in state
    """

    def __init__(self, key: str, available_keys: list[str]):
        """
        Initialize StateKeyError with context.

        Args:
            key: Key that was not found
            available_keys: List of available keys in state
        """
        self.key = key
        self.available_keys = available_keys

        # Format available keys for display (show max 10)
        if len(available_keys) <= 10:
            keys_display = ", ".join(available_keys)
        else:
            keys_display = ", ".join(available_keys[:10]) + "..."

        super().__init__(
            f"Key '{key}' not found in state. " f"Available keys: {keys_display}"
        )


class StateVersionError(ThreadSafeStateError, ValueError):
    """
    Raised when version-related operations fail.

    Attributes:
        current_version: Current state version
        target_version: Target version that caused the error
    """

    def __init__(self, message: str, current_version: int, target_version: int):
        """
        Initialize StateVersionError with version context.

        Args:
            message: Error message
            current_version: Current state version
            target_version: Target version
        """
        self.current_version = current_version
        self.target_version = target_version
        super().__init__(
            f"{message} (current: {current_version}, target: {target_version})"
        )


class StateRollbackError(ThreadSafeStateError):
    """
    Raised when rollback operation fails.

    Attributes:
        target_version: Version to rollback to
        oldest_version: Oldest version available in history
    """

    def __init__(
        self, message: str, target_version: int, oldest_version: Optional[int] = None
    ):
        """
        Initialize StateRollbackError with rollback context.

        Args:
            message: Error message
            target_version: Target version for rollback
            oldest_version: Oldest version available in history
        """
        self.target_version = target_version
        self.oldest_version = oldest_version
        super().__init__(
            f"{message} (target: {target_version}, oldest: {oldest_version})"
        )


class StateLockTimeoutError(ThreadSafeStateError, TimeoutError):
    """
    Raised when lock acquisition times out.

    Note: This is a future enhancement for optional lock timeouts.

    Attributes:
        timeout: Timeout duration in seconds
        operation: Operation that timed out
    """

    def __init__(self, timeout: float, operation: str):
        """
        Initialize StateLockTimeoutError.

        Args:
            timeout: Timeout duration in seconds
            operation: Operation that timed out
        """
        self.timeout = timeout
        self.operation = operation
        super().__init__(
            f"Lock acquisition timeout after {timeout}s for operation: {operation}"
        )


class DependencyResolutionError(Exception):
    """
    Base exception for dependency resolution errors.

    Attributes:
        coordination_id: Coordination session ID
        dependency_id: Dependency identifier
        error_message: Detailed error message
    """

    def __init__(
        self,
        coordination_id: str,
        dependency_id: str,
        error_message: str,
    ):
        """
        Initialize DependencyResolutionError.

        Args:
            coordination_id: Coordination session ID
            dependency_id: Dependency identifier
            error_message: Detailed error message
        """
        self.coordination_id = coordination_id
        self.dependency_id = dependency_id
        self.error_message = error_message
        super().__init__(
            f"Dependency resolution failed for dependency '{dependency_id}' "
            f"in coordination session '{coordination_id}': {error_message}"
        )


class DependencyTimeoutError(DependencyResolutionError, TimeoutError):
    """
    Raised when dependency resolution times out.

    Attributes:
        coordination_id: Coordination session ID
        dependency_id: Dependency identifier
        timeout_seconds: Timeout duration in seconds
    """

    def __init__(
        self,
        coordination_id: str,
        dependency_id: str,
        timeout_seconds: int,
    ):
        """
        Initialize DependencyTimeoutError.

        Args:
            coordination_id: Coordination session ID
            dependency_id: Dependency identifier
            timeout_seconds: Timeout duration in seconds
        """
        self.timeout_seconds = timeout_seconds
        error_message = f"Timeout after {timeout_seconds}s"
        super().__init__(coordination_id, dependency_id, error_message)
