# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Domain plugin result model.

This module provides the ModelDomainPluginResult dataclass for representing
the outcome of domain plugin lifecycle operations.

Design Pattern:
    Result models provide structured information about operation outcomes,
    including success/failure status, timing, and resources created. This
    model supports the custom __bool__ pattern where truthiness reflects
    operation success.

Thread Safety:
    ModelDomainPluginResult is NOT frozen - it may contain mutable lists
    of resources and callbacks.

Example:
    >>> from omnibase_infra.runtime.models import ModelDomainPluginResult
    >>>
    >>> # Create success result using factory method
    >>> result = ModelDomainPluginResult.succeeded(
    ...     plugin_id="registration",
    ...     message="Plugin initialized",
    ...     resources_created=["postgres_pool"],
    ... )
    >>> if result:  # Uses __bool__ - True if success
    ...     print("Initialization complete!")
    >>>
    >>> # Create failure result
    >>> failure = ModelDomainPluginResult.failed(
    ...     plugin_id="registration",
    ...     error_message="Connection refused",
    ... )
    >>> if not failure:
    ...     print(f"Failed: {failure.error_message}")

Related:
    - OMN-1346: Registration Code Extraction
    - OMN-888: Registration Orchestrator
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class ModelDomainPluginResult:
    """Result model returned by domain plugin lifecycle hooks.

    Provides structured information about the outcome of a plugin operation,
    including success/failure status, timing, and resources created or cleaned up.

    Attributes:
        plugin_id: Identifier of the plugin that produced this result.
        success: Whether the operation succeeded.
        message: Human-readable message describing the outcome.
        resources_created: List of resource identifiers created during this operation.
        services_registered: List of service class names registered in container.
        duration_seconds: Time taken for the operation.
        error_message: Error message if operation failed.
        unsubscribe_callbacks: Callbacks to invoke during shutdown for cleanup.

    Example:
        ```python
        result = ModelDomainPluginResult(
            plugin_id="registration",
            success=True,
            message="Registration plugin initialized",
            resources_created=["postgres_pool"],
            services_registered=["HandlerNodeIntrospected", "HandlerRuntimeTick"],
            duration_seconds=0.5,
        )
        if not result:
            logger.error("Plugin failed: %s", result.error_message)
        ```
    """

    plugin_id: str
    success: bool
    message: str = ""
    resources_created: list[str] = field(default_factory=list)
    services_registered: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: str | None = None

    # Store unsubscribe callbacks for cleanup
    unsubscribe_callbacks: list[Callable[[], Awaitable[None]]] = field(
        default_factory=list
    )

    def get_error_message_or_default(self, default: str = "unknown") -> str:
        """Return error_message if set, otherwise the default value.

        This helper simplifies error handling in plugin activation code,
        avoiding verbose None checks when accessing error_message.

        Args:
            default: Value to return if error_message is None or empty.

        Returns:
            The error_message if it's a non-empty string, otherwise default.

        Example:
            >>> result = ModelDomainPluginResult.failed(
            ...     plugin_id="test",
            ...     error_message="Connection refused",
            ... )
            >>> result.get_error_message_or_default()
            'Connection refused'
            >>>
            >>> success = ModelDomainPluginResult.succeeded(plugin_id="test")
            >>> success.get_error_message_or_default("no error")
            'no error'
        """
        return self.error_message if self.error_message else default

    def __bool__(self) -> bool:
        """Return True if the operation succeeded.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when ``success`` is True. This differs from typical
            Pydantic/dataclass behavior where ``bool(model)`` always returns ``True``
            for any valid instance.

            This design enables idiomatic plugin lifecycle checks::

                result = await plugin.initialize(config)
                if result:
                    # Operation succeeded - continue
                    await plugin.wire_handlers(config)
                else:
                    # Operation failed - handle error
                    logger.error("Plugin failed: %s", result.error_message)

            If you need to check instance validity instead, use explicit checks::

                # Check for success (uses __bool__)
                if result:
                    ...

                # Check instance is valid (always True for constructed instance)
                if result is not None:
                    ...

                # Explicit success check (preferred for clarity)
                if result.success:
                    ...

        Returns:
            True if success is True, False otherwise.
        """
        return self.success

    @classmethod
    def succeeded(
        cls,
        plugin_id: str,
        message: str = "",
        duration_seconds: float = 0.0,
    ) -> ModelDomainPluginResult:
        """Create a simple success result.

        For results that require additional fields (resources_created, services_registered,
        unsubscribe_callbacks), use the constructor directly.

        Args:
            plugin_id: Identifier of the plugin.
            message: Human-readable success message.
            duration_seconds: Time taken for the operation.

        Returns:
            ModelDomainPluginResult with success=True.

        Example:
            >>> result = ModelDomainPluginResult.succeeded(
            ...     plugin_id="registration",
            ...     message="Initialized",
            ... )
            >>> result.success
            True

            >>> # For results with resources, use constructor:
            >>> result = ModelDomainPluginResult(
            ...     plugin_id="registration",
            ...     success=True,
            ...     message="Initialized",
            ...     resources_created=["pool"],
            ... )
        """
        return cls(
            plugin_id=plugin_id,
            success=True,
            message=message,
            duration_seconds=duration_seconds,
        )

    @classmethod
    def failed(
        cls,
        plugin_id: str,
        error_message: str,
        message: str = "",
        duration_seconds: float = 0.0,
    ) -> ModelDomainPluginResult:
        """Create a failure result.

        Args:
            plugin_id: Identifier of the plugin.
            error_message: Description of the error that occurred.
            message: Optional human-readable message (defaults to "Plugin {id} failed").
            duration_seconds: Time taken before failure.

        Returns:
            ModelDomainPluginResult with success=False.

        Example:
            >>> result = ModelDomainPluginResult.failed(
            ...     plugin_id="registration",
            ...     error_message="Connection refused",
            ... )
            >>> result.success
            False
            >>> result.error_message
            'Connection refused'
        """
        return cls(
            plugin_id=plugin_id,
            success=False,
            message=message or f"Plugin {plugin_id} failed",
            error_message=error_message,
            duration_seconds=duration_seconds,
        )

    @classmethod
    def skipped(
        cls,
        plugin_id: str,
        reason: str,
    ) -> ModelDomainPluginResult:
        """Create a skipped result (plugin did not activate).

        Args:
            plugin_id: Identifier of the plugin.
            reason: Explanation of why the plugin was skipped.

        Returns:
            ModelDomainPluginResult with success=True and skip message.

        Example:
            >>> result = ModelDomainPluginResult.skipped(
            ...     plugin_id="registration",
            ...     reason="POSTGRES_HOST not configured",
            ... )
            >>> result.success
            True
            >>> "skipped" in result.message
            True
        """
        return cls(
            plugin_id=plugin_id,
            success=True,
            message=f"Plugin {plugin_id} skipped: {reason}",
        )


__all__ = ["ModelDomainPluginResult"]
