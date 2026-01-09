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

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelFileSystemConfig(BaseModel):
    """Configuration for HandlerFileSystem initialization.

    This model defines the configuration parameters required to initialize
    the filesystem handler, including the security-critical path whitelist
    and optional size limits.

    Attributes:
        allowed_paths: List of directory paths that the handler is allowed
            to access. This is a required security boundary - operations
            on paths outside these directories will be rejected.
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
        ['/tmp/test', '/data/output']
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    allowed_paths: list[str] = Field(
        min_length=1,
        description="List of allowed directory paths for filesystem operations. "
        "Required for security - must not be empty.",
    )
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
