# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Handler Configuration Model.

This module provides the Pydantic model for HandlerFileSystem initialization
configuration, replacing the untyped dict[str, object] pattern.

Security:
    The allowed_paths field defines the whitelist of directories that the
    filesystem handler can access. This is a critical security boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelFileSystemConfig(BaseModel):
    """Configuration for HandlerFileSystem initialization.

    This model defines the configuration parameters required to initialize
    the filesystem handler, including the security-critical path whitelist
    and optional size limits.

    Attributes:
        allowed_paths: Immutable tuple of directory paths that the handler is
            allowed to access. This is a required security boundary - operations
            on paths outside these directories will be rejected. Uses tuple
            instead of list to prevent runtime mutation of the whitelist.
        max_read_size: Maximum file size in bytes for read operations.
            If None, the default from environment or code will be used.
        max_write_size: Maximum content size in bytes for write operations.
            If None, the default from environment or code will be used.
        correlation_id: Optional correlation ID for initialization tracing.

    Example:
        >>> config = ModelFileSystemConfig(
        ...     allowed_paths=["/tmp/test", "/data/output"],
        ...     max_read_size=10 * 1024 * 1024,  # 10 MB
        ... )
        >>> print(config.allowed_paths)
        ('/tmp/test', '/data/output')
        >>> # List input is automatically coerced to immutable tuple
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    allowed_paths: tuple[str, ...] = Field(
        min_length=1,
        description="Immutable tuple of allowed directory paths for filesystem operations. "
        "Required for security - must not be empty. List inputs are coerced to tuple.",
    )

    @field_validator("allowed_paths", mode="before")
    @classmethod
    def coerce_to_tuple(cls, v: Sequence[str]) -> tuple[str, ...]:
        """Coerce list inputs to immutable tuple for security.

        This ensures the allowed_paths whitelist cannot be mutated after
        model construction, even though Python's frozen=True only prevents
        reassignment of the attribute, not mutation of mutable objects.

        Args:
            v: Input value, either a list or tuple of strings.

        Returns:
            Immutable tuple of path strings.
        """
        # Always return tuple to satisfy return type
        return tuple(v)

    max_read_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum file size in bytes for read operations. "
        "If None, uses default (100 MB from env or code).",
    )
    max_write_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum content size in bytes for write operations. "
        "If None, uses default (50 MB from env or code).",
    )
    correlation_id: UUID | str | None = Field(
        default=None,
        description="Optional correlation ID for initialization tracing.",
    )


__all__: list[str] = ["ModelFileSystemConfig"]
