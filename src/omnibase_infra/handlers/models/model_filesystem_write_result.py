# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Write File Result Model.

This module provides the Pydantic model for the filesystem.write_file operation
result, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelWriteFileResult(BaseModel):
    """Result from filesystem.write_file operation.

    Contains metadata about the completed write operation.

    Attributes:
        path: The resolved absolute path to the file that was written.
        bytes_written: Number of bytes written to the file.
        created: True if a new file was created, False if an existing
            file was overwritten.

    Example:
        >>> result = ModelWriteFileResult(
        ...     path="/data/output.txt",
        ...     bytes_written=13,
        ...     created=True,
        ... )
        >>> print(result.created)
        True
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    path: str = Field(
        min_length=1,
        description="Resolved absolute path to the file.",
    )
    bytes_written: int = Field(
        ge=0,
        description="Number of bytes written to the file.",
    )
    created: bool = Field(
        description="True if a new file was created, False if overwritten.",
    )


__all__: list[str] = ["ModelWriteFileResult"]
