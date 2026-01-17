# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for cached secret entries.

.. versionadded:: 0.8.0
    Initial implementation for OMN-764.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from omnibase_infra.runtime.models.model_secret_source_spec import SecretSourceType


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
    source_type: SecretSourceType = Field(
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

    @field_validator("cached_at", "expires_at", mode="before")
    @classmethod
    def ensure_utc_aware(cls, v: object) -> object:
        """Ensure datetime fields are timezone-aware (UTC).

        This validator prevents naive/aware datetime comparison errors in
        is_expired() by ensuring all datetime fields are UTC-aware.

        Args:
            v: The input value. In "before" mode, this can be any type.
               Non-datetime values are passed through for Pydantic to validate.

        Returns:
            UTC-aware datetime if input is datetime, otherwise unchanged input.
            Naive datetimes are treated as UTC. Non-UTC timezones are converted.

        Note:
            - Non-datetime inputs are returned unchanged for Pydantic's type validation
            - Naive datetimes are treated as UTC (v.replace(tzinfo=UTC))
            - Non-UTC timezones are converted to UTC (v.astimezone(UTC))
            - Uses utcoffset() comparison for robust UTC detection across different
              timezone implementations (timezone.utc, datetime.UTC, pytz.UTC, etc.)
        """
        if not isinstance(v, datetime):
            return v  # Let Pydantic handle type validation
        if v.tzinfo is None:
            # Treat naive datetime as UTC
            return v.replace(tzinfo=UTC)
        # Check if already UTC by comparing offset (handles different UTC representations)
        if v.utcoffset() == timedelta(0):
            # Already UTC (or equivalent), normalize to datetime.UTC
            if v.tzinfo is not UTC:
                return v.replace(tzinfo=UTC)
            return v
        # Non-UTC timezone, convert to UTC
        return v.astimezone(UTC)

    def is_expired(self) -> bool:
        """Check if this cached entry has expired.

        Returns:
            True if the current UTC time is past the expiration time.

        Note:
            The field validator ensures expires_at is always UTC-aware,
            so this comparison is always safe.

        Example:
            >>> cached = ModelCachedSecret(...)
            >>> if cached.is_expired():
            ...     # Refresh the secret
            ...     pass
        """
        # expires_at is guaranteed UTC-aware by field validator
        return datetime.now(UTC) > self.expires_at


__all__: list[str] = ["ModelCachedSecret"]
