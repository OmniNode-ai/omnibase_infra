# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for BindingConfigResolver.

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.

This module provides the Pydantic configuration model for the BindingConfigResolver,
which resolves handler-specific configuration from multiple sources (file, env, vault).

Example:
    >>> from pathlib import Path
    >>> from omnibase_infra.runtime.models import ModelBindingConfigResolverConfig
    >>> config = ModelBindingConfigResolverConfig(
    ...     config_dir=Path("/workspace/configs"),
    ...     cache_ttl_seconds=600.0,
    ...     env_prefix="HANDLER",
    ... )
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelBindingConfigResolverConfig(BaseModel):
    """Configuration for BindingConfigResolver.

    Configures the binding configuration resolution system that supports
    multiple configuration sources with priority-based resolution.

    Supported config_ref Schemes:
        - vault: - Vault secrets (e.g., vault:secret/data/db#password)
        - env: - Environment variables (e.g., env:DB_CONFIG_JSON)
        - file: - File-based configuration (e.g., file:configs/db.yaml)

    Note:
        The config_ref scheme determines WHERE to load base config from,
        not priority between schemes. Only one config_ref is used per call.

    Environment Variable Override Pattern:
        When env_prefix is set (e.g., "HANDLER"), the resolver looks for:
        {env_prefix}_{HANDLER_TYPE}_{FIELD} (e.g., HANDLER_DB_POOL_SIZE)

    Attributes:
        config_dir: Base directory for relative file: paths.
            If None, file: paths must be absolute.
        enable_caching: Whether to cache resolved configurations.
        cache_ttl_seconds: Time-to-live for cached configurations (0-86400).
        env_prefix: Prefix for environment variable overrides.
            Pattern: {env_prefix}_{HANDLER_TYPE}_{FIELD}
        strict_validation: If True, fail on unknown fields in resolved config.
            If False, ignore unknown fields.
        allowed_schemes: Set of allowed config_ref URI schemes for security.
            Only these schemes can be used in config_ref values.

    Note:
        SecretResolver is resolved from the container via dependency injection
        in BindingConfigResolver.__init__, following ONEX mandatory container
        injection pattern per CLAUDE.md.

    Example:
        >>> config = ModelBindingConfigResolverConfig(
        ...     config_dir=Path("/etc/onex/handlers"),
        ...     cache_ttl_seconds=300.0,
        ...     env_prefix="ONEX_HANDLER",
        ...     strict_validation=True,
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Base configuration directory for relative file: paths
    config_dir: Path | None = Field(
        default=None,
        description="Base directory for resolving relative file: paths. "
        "If None, all file: paths must be absolute.",
    )

    # Cache settings
    enable_caching: bool = Field(
        default=True,
        description="Whether to cache resolved configurations. "
        "Disable for development or when configurations change frequently.",
    )
    cache_ttl_seconds: float = Field(
        default=300.0,
        ge=0.0,
        le=86400.0,
        description="Time-to-live for cached configurations in seconds (0-86400). "
        "Default is 5 minutes.",
    )

    # Environment variable override settings
    env_prefix: str = Field(
        default="HANDLER",
        description="Prefix for environment variable overrides. "
        "Pattern: {env_prefix}_{HANDLER_TYPE}_{FIELD}. "
        "Must be a valid Python identifier.",
    )

    # NOTE: SecretResolver is now resolved from container via dependency injection.
    # See BindingConfigResolver.__init__ which resolves SecretResolver from
    # container.service_registry.resolve_service(SecretResolver).
    # This follows ONEX mandatory container injection pattern per CLAUDE.md.

    # Validation strictness
    strict_validation: bool = Field(
        default=True,
        description="If True, fail on unknown fields in resolved configuration. "
        "If False, ignore unknown fields.",
    )

    # Environment variable type coercion strictness
    strict_env_coercion: bool = Field(
        default=False,
        description="If True, raise ProtocolConfigurationError when environment "
        "variable values cannot be converted to the expected type. "
        "If False, log a warning and skip the override. "
        "Default is False for backwards compatibility.",
    )

    # Allowed config_ref schemes (for security)
    allowed_schemes: frozenset[str] = Field(
        default=frozenset({"file", "env", "vault"}),
        description="Set of allowed config_ref URI schemes for security. "
        "Only these schemes can be used in config_ref values.",
    )

    # Vault error handling behavior
    fail_on_vault_error: bool = Field(
        default=False,
        description="If True, raise ProtocolConfigurationError when a vault: "
        "reference fails to resolve. If False, log an error and keep the original "
        "placeholder value (which may be insecure). "
        "Set to True in production to prevent silent security fallbacks.",
    )

    @field_validator("env_prefix")
    @classmethod
    def validate_env_prefix(cls, value: str) -> str:
        """Validate env_prefix is a valid identifier.

        The prefix must be a valid Python identifier (alphanumeric and underscores,
        not starting with a digit) to ensure it can be used in environment variable
        names.

        Args:
            value: The environment variable prefix.

        Returns:
            The validated prefix (uppercase).

        Raises:
            ValueError: If prefix is not a valid identifier.
        """
        if not value:
            msg = "env_prefix cannot be empty"
            raise ValueError(msg)

        # Check if it's a valid Python identifier
        # Must match: [a-zA-Z_][a-zA-Z0-9_]*
        identifier_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        if not identifier_pattern.match(value):
            msg = (
                f"env_prefix must be a valid identifier "
                f"(alphanumeric and underscores, not starting with digit), got '{value}'"
            )
            raise ValueError(msg)

        # Return uppercase for consistency with environment variable conventions
        return value.upper()

    @field_validator("config_dir")
    @classmethod
    def validate_config_dir(cls, value: Path | None) -> Path | None:
        """Validate config_dir exists if provided.

        Args:
            value: The configuration directory path, or None.

        Returns:
            The validated path, or None.

        Raises:
            ValueError: If path is provided but does not exist or is not a directory.
        """
        if value is None:
            return None

        if not value.exists():
            msg = f"config_dir does not exist: {value}"
            raise ValueError(msg)

        if not value.is_dir():
            msg = f"config_dir is not a directory: {value}"
            raise ValueError(msg)

        return value

    @field_validator("allowed_schemes")
    @classmethod
    def validate_allowed_schemes(cls, value: frozenset[str]) -> frozenset[str]:
        """Validate allowed_schemes contains only recognized schemes.

        Args:
            value: The set of allowed URI schemes.

        Returns:
            The validated scheme set.

        Raises:
            ValueError: If set is empty or contains unrecognized schemes.
        """
        if not value:
            msg = "allowed_schemes cannot be empty"
            raise ValueError(msg)

        # Known valid schemes
        valid_schemes = {"file", "env", "vault"}
        unknown_schemes = value - valid_schemes

        if unknown_schemes:
            msg = (
                f"allowed_schemes contains unrecognized schemes: {unknown_schemes}. "
                f"Valid schemes are: {valid_schemes}"
            )
            raise ValueError(msg)

        return value


__all__: list[str] = ["ModelBindingConfigResolverConfig"]
