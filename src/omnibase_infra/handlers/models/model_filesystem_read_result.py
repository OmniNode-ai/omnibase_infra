# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Read File Result Model.

This module provides the Pydantic model for the filesystem.read_file operation
result, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelReadFileResult(BaseModel):
    """Result from filesystem.read_file operation.

    Contains the file content and metadata about the read operation.

    Attributes:
        content: The file content. For binary=True, this is a base64-encoded
            string. For binary=False, this is the decoded text content.
        size: Size of the file in bytes (original size, not encoded size).
        path: The resolved absolute path to the file that was read.
        binary: Whether the file was read in binary mode.

    Example:
        >>> result = ModelReadFileResult(
        ...     content="Hello, World!",
        ...     size=13,
        ...     path="/data/greeting.txt",
        ...     binary=False,
        ... )
        >>> print(result.size)
        13
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    content: str = Field(
        description="File content. Base64-encoded for binary mode, "
        "text for non-binary mode.",
    )
    size: int = Field(
        ge=0,
        description="File size in bytes.",
    )
    path: str = Field(
        min_length=1,
        description="Resolved absolute path to the file.",
    )
    binary: bool = Field(
        description="Whether the file was read in binary mode.",
    )


__all__: list[str] = ["ModelReadFileResult"]
