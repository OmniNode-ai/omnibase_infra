# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for secret source information (introspection).

.. versionadded:: 0.8.0
    Initial implementation for OMN-764.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_secret_source_spec import SecretSourceType


class ModelSecretSourceInfo(BaseModel):
    """Non-sensitive source information for introspection.

    Provides metadata about a secret's source without exposing the
    actual secret value. Used for debugging and observability.

    Attributes:
        logical_name: The logical name of the secret.
        source_type: The source type from which the secret is resolved.
        source_path_masked: Masked version of the source path for logging.
        is_cached: Whether the secret is currently cached.
        expires_at: Cache expiration time if cached, None otherwise.

    Example:
        >>> info = ModelSecretSourceInfo(
        ...     logical_name="database.password",
        ...     source_type="vault",
        ...     source_path_masked="vault:secret/data/***",
        ...     is_cached=True,
        ...     expires_at=datetime.utcnow(),
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
        description="The logical name of the secret.",
    )
    source_type: SecretSourceType = Field(
        ...,
        description="The source type from which the secret is resolved.",
    )
    source_path_masked: str = Field(
        ...,
        description="Masked source path safe for logging "
        "(e.g., 'env:POSTGRES_***' or 'vault:secret/data/***').",
    )
    is_cached: bool = Field(
        ...,
        description="Whether the secret is currently in the cache.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Cache expiration timestamp if cached, None if not cached.",
    )


__all__: list[str] = ["ModelSecretSourceInfo"]
