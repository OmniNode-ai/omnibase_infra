# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for SecretResolver.

.. versionadded:: 0.8.0
    Initial implementation for OMN-764.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping


class ModelSecretResolverConfig(BaseModel):
    """Configuration for SecretResolver.

    Configures the centralized secret resolution system that supports
    multiple secret sources with priority-based resolution.

    Source Priority Order:
        1. Vault (if configured) - for production secrets
        2. Environment variables - for local development
        3. File-based secrets - for Kubernetes deployments

    Attributes:
        mappings: Explicit mappings from logical names to source specs.
        default_ttl_env_seconds: Default TTL for environment variable secrets.
        default_ttl_file_seconds: Default TTL for file-based secrets.
        default_ttl_vault_seconds: Default TTL for Vault secrets.
        enable_convention_fallback: Enable automatic source discovery by convention.
        convention_env_prefix: Prefix for environment variable convention lookup.
        bootstrap_secrets: Secrets resolved directly from env (never through chain).
        secrets_dir: Directory for file-based secrets (K8s secrets volume).

    Example:
        >>> config = ModelSecretResolverConfig(
        ...     default_ttl_vault_seconds=60,
        ...     enable_convention_fallback=True,
        ...     convention_env_prefix="ONEX_",
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,  # Allow mutation for runtime configuration
        extra="forbid",
        from_attributes=True,
    )

    # Explicit mappings: logical_name -> source_spec
    mappings: list[ModelSecretMapping] = Field(
        default_factory=list,
        description="Explicit mappings from logical secret names to source specifications. "
        "Takes precedence over convention-based resolution.",
    )

    # Default TTLs by source type (in seconds)
    default_ttl_env_seconds: int = Field(
        default=86400,
        ge=0,
        description="Default TTL for environment variable secrets (24 hours).",
    )
    default_ttl_file_seconds: int = Field(
        default=86400,
        ge=0,
        description="Default TTL for file-based secrets (24 hours).",
    )
    default_ttl_vault_seconds: int = Field(
        default=300,
        ge=0,
        description="Default TTL for Vault secrets (5 minutes).",
    )

    # Convention fallback when no explicit mapping exists
    enable_convention_fallback: bool = Field(
        default=True,
        description="Enable automatic source discovery using naming conventions. "
        "When True, 'database.postgres.password' becomes 'DATABASE_POSTGRES_PASSWORD'.",
    )
    convention_env_prefix: str = Field(
        default="",
        description="Prefix added to environment variable names during convention lookup. "
        "E.g., 'ONEX_' makes 'database.password' look for 'ONEX_DATABASE_PASSWORD'.",
    )

    # Bootstrap secrets (always resolved from env, never through resolver chain)
    bootstrap_secrets: list[str] = Field(
        default_factory=lambda: ["vault.token", "vault.addr", "vault.ca_cert"],
        description="Secrets that are always resolved directly from environment variables, "
        "never through the resolver chain. Used for Vault bootstrap credentials.",
    )

    # File-based secrets directory (K8s secrets volume mount)
    secrets_dir: Path = Field(
        default=Path("/run/secrets"),
        description="Directory containing file-based secrets (K8s secrets volume mount).",
    )

    # NOTE: Vault configuration will be added when ModelVaultHandlerConfig is available
    # vault_config: ModelVaultHandlerConfig | None = None


__all__: list[str] = ["ModelSecretResolverConfig"]
