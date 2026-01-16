# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for cached secret entries.

.. versionadded:: 0.8.0
    Initial implementation for OMN-764.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModelCachedSecret(BaseModel):
    """Cached secret with TTL tracking.

    Represents a secret value that has been resolved and cached,
    with metadata for cache invalidation and observability.

    Attributes:
        value: The secret value (masked in logs and repr).
        source_type: The source from which the secret was resolved.
        logical_name: The logical name used to request the secret.
        cached_at: Timestamp when the secret was cached.
        expires_at: Timestamp when the cached entry expires.
        hit_count: Number of cache hits for this entry.

    Example:
        >>> from datetime import UTC, datetime, timedelta
        >>> from pydantic import SecretStr
        >>>
        >>> cached = ModelCachedSecret(
        ...     value=SecretStr("password123"),
        ...     source_type="env",
        ...     logical_name="db.password",
        ...     cached_at=datetime.now(UTC),
        ...     expires_at=datetime.now(UTC) + timedelta(hours=24),
        ... )
        >>> cached.is_expired()
        False
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    value: SecretStr = Field(
        ...,
        description="The cached secret value. Automatically masked in logs and repr.",
    )
    source_type: Literal["env", "vault", "file"] = Field(
        ...,
        description="The source type from which this secret was resolved.",
    )
    logical_name: str = Field(
        ...,
        min_length=1,
        description="The logical name used to request this secret.",
    )
    cached_at: datetime = Field(
        ...,
        description="UTC timestamp when the secret was cached.",
    )
    expires_at: datetime = Field(
        ...,
        description="UTC timestamp when this cached entry expires.",
    )
    hit_count: int = Field(
        default=0,
        ge=0,
        description="Number of cache hits for this entry since caching.",
    )

    def is_expired(self) -> bool:
        """Check if this cached entry has expired.

        Returns:
            True if the current UTC time is past the expiration time.

        Note:
            This method handles both timezone-aware and timezone-naive
            datetimes. Naive datetimes are treated as UTC.

        Example:
            >>> cached = ModelCachedSecret(...)
            >>> if cached.is_expired():
            ...     # Refresh the secret
            ...     pass
        """
        now = datetime.now(UTC)
        expires = self.expires_at
        if expires.tzinfo is None:
            # Treat naive datetime as UTC
            expires = expires.replace(tzinfo=UTC)
        return now > expires


__all__: list[str] = ["ModelCachedSecret"]
