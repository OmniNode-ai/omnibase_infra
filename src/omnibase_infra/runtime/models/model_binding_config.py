# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Binding configuration model for BindingConfigResolver.

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.

This module provides the configuration model for handler bindings used
by BindingConfigResolver.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.types import JsonType
from omnibase_infra.runtime.models.model_retry_policy import ModelRetryPolicy


class ModelBindingConfig(BaseModel):
    """Configuration for binding a handler to the runtime.

    Defines how a specific handler is configured, including its operational
    parameters, retry behavior, and optional external configuration references.

    Attributes:
        handler_type: Handler type identifier (e.g., "vault", "db", "consul").
            Used for handler discovery and routing.
        name: Optional display name for the handler. Defaults to handler_type
            if not specified.
        enabled: Whether the handler is active. Disabled handlers are not loaded.
        priority: Execution priority for handler ordering. Higher values execute
            first. Use negative values to ensure late execution.
        config_ref: Reference to external configuration. Supported schemes:
            - "file:configs/handler.yaml" - File path (preferred format)
            - "file:/absolute/path/config.yaml" - Absolute file path
            - "env:CONFIG_VAR" - Load from environment variable
            - "vault:secret/data/path" - Load from Vault secret
            - "vault:secret/data/path#field" - Load specific field from Vault secret
            Note: file:// prefix is also supported for backwards compatibility.
        config: Inline configuration dictionary. If both config_ref and config
            are provided, config takes precedence for overlapping keys.
        retry_policy: Retry configuration for transient failures.
        timeout_ms: Operation timeout in milliseconds. Applies to individual
            handler operations, not total request time.
        rate_limit_per_second: Maximum operations per second. None means no limit.

    Example:
        >>> config = ModelBindingConfig(
        ...     handler_type="db",
        ...     name="primary-postgres",
        ...     priority=10,
        ...     timeout_ms=5000,
        ...     retry_policy=ModelRetryPolicy(max_retries=5),
        ...     rate_limit_per_second=100.0,
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    handler_type: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Handler type identifier (e.g., 'vault', 'db', 'consul'). "
        "Must be non-empty and match a registered handler implementation.",
    )

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional display name for the handler. "
        "Defaults to handler_type if not specified.",
    )

    enabled: bool = Field(
        default=True,
        description="Whether this handler binding is active. "
        "Disabled handlers are skipped during runtime initialization.",
    )

    priority: int = Field(
        default=0,
        ge=-100,
        le=100,
        description="Execution priority for handler ordering (-100 to 100). "
        "Higher values execute first. Use 0 for default ordering.",
    )

    config_ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Reference to external configuration. "
        "Supported schemes: file: (including file://), env:, vault:",
    )

    config: dict[str, JsonType] | None = Field(
        default=None,
        description="Inline configuration dictionary. "
        "Takes precedence over config_ref for overlapping keys.",
    )

    retry_policy: ModelRetryPolicy | None = Field(
        default=None,
        description="Retry configuration for transient failures. "
        "If not specified, handler uses its default retry behavior.",
    )

    timeout_ms: int = Field(
        default=30000,
        ge=100,
        le=600000,
        description="Operation timeout in milliseconds (100-600000). "
        "Default 30s; max 10min for long-running operations.",
    )

    rate_limit_per_second: float | None = Field(
        default=None,
        ge=0.1,
        le=10000.0,
        description="Maximum operations per second (0.1-10000). "
        "None means no rate limiting.",
    )

    @field_validator("config_ref")
    @classmethod
    def validate_config_ref_scheme(cls, v: str | None) -> str | None:
        """Validate that config_ref uses a supported scheme.

        Args:
            v: The config_ref value to validate.

        Returns:
            The validated config_ref value.

        Raises:
            ValueError: If config_ref uses an unsupported scheme.
        """
        if v is None:
            return v

        # Use "file:" to accept both "file://path" and shorthand "file:path" formats
        supported_schemes = ("file:", "env:", "vault:")
        if not any(v.startswith(scheme) for scheme in supported_schemes):
            raise ValueError(
                f"config_ref must start with one of {supported_schemes}, got: {v!r}"
            )
        return v

    def get_effective_name(self) -> str:
        """Get the effective name for this handler binding.

        Returns:
            The name field if set, otherwise the handler_type.
        """
        return self.name if self.name is not None else self.handler_type


__all__: list[str] = [
    "ModelBindingConfig",
]
