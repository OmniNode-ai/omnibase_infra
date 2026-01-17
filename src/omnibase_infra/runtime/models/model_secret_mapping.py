# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for secret mapping configuration.

.. versionadded:: 0.8.0
    Initial implementation for OMN-764.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_secret_source_spec import ModelSecretSourceSpec


class ModelSecretMapping(BaseModel):
    """Mapping from logical name to source specification.

    Defines an explicit mapping between a logical secret name used in the
    application and its concrete source location.

    Attributes:
        logical_name: Application-level secret identifier using dot notation.
        source: Specification of where to retrieve the secret.
        ttl_seconds: Optional TTL override for this specific secret.

    Example:
        >>> mapping = ModelSecretMapping(
        ...     logical_name="database.postgres.password",
        ...     source=ModelSecretSourceSpec(
        ...         source_type="vault",
        ...         source_path="secret/data/database/postgres#password",
        ...     ),
        ...     ttl_seconds=60,  # Refresh every minute
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    logical_name: str = Field(
        ...,
        min_length=1,
        description="Logical name for the secret using dot notation "
        "(e.g., 'database.postgres.password').",
    )
    source: ModelSecretSourceSpec = Field(
        ...,
        description="Source specification defining where to retrieve the secret.",
    )
    ttl_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Optional TTL override in seconds for this specific secret. "
        "If None, uses the default TTL for the source type.",
    )


__all__: list[str] = ["ModelSecretMapping"]
