# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Compute Registry Error Class.

This module defines the ComputeRegistryError for compute registry operations.
"""

from typing import Any, cast

from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class ComputeRegistryError(RuntimeHostError):
    """Error raised when compute registry operations fail.

    Used for:
    - Attempting to get an unregistered compute plugin
    - Registration failures (async validation without flag, protocol validation)
    - Plugin validation failures during registration
    - Invalid semver format

    Extends RuntimeHostError as this is an infrastructure-layer runtime concern.

    Example:
        >>> from omnibase_infra.errors import ComputeRegistryError
        >>> from omnibase_infra.errors import ModelInfraErrorContext
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> from uuid import uuid4

        >>> # Plugin not found
        >>> try:
        ...     plugin = registry.get("unknown_plugin")
        ... except ComputeRegistryError as e:
        ...     print(f"Plugin not found: {e}")

        >>> # With context
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="get_plugin",
        ...     correlation_id=uuid4(),
        ... )
        >>> raise ComputeRegistryError(
        ...     "Compute plugin not registered",
        ...     plugin_id="json_normalizer",
        ...     version="1.0.0",
        ...     context=context,
        ... )

        >>> # Async validation failure
        >>> raise ComputeRegistryError(
        ...     "Plugin has async execute() but deterministic_async=True not specified",
        ...     plugin_id="async_transformer",
        ...     async_method="execute",
        ... )
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        version: str | None = None,
        registered_plugins: list[str] | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ComputeRegistryError.

        Args:
            message: Human-readable error message
            plugin_id: The plugin ID that caused the error (if applicable)
            version: The version that caused the error (if applicable)
            registered_plugins: List of currently registered plugin IDs (for error context)
            context: Bundled infrastructure context for correlation_id and structured fields
            **extra_context: Additional context information (e.g., async_method)
        """
        # Add plugin_id, version, registered_plugins to extra_context if provided
        if plugin_id is not None:
            extra_context["plugin_id"] = plugin_id
        if version is not None:
            extra_context["version"] = version
        if registered_plugins is not None:
            extra_context["registered_plugins"] = registered_plugins

        # NOTE: Cast required for mypy - **dict[str, object] doesn't satisfy **context: Any
        super().__init__(
            message=message,
            context=context,
            **cast("dict[str, Any]", extra_context),
        )


__all__ = ["ComputeRegistryError"]
