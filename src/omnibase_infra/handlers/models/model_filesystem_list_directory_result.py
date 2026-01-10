# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem List Directory Result Model.

This module provides the Pydantic model for the filesystem.list_directory operation
result, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.models.model_filesystem_directory_entry import (
    ModelDirectoryEntry,
)


class ModelListDirectoryResult(BaseModel):
    """Result from filesystem.list_directory operation.

    Contains the directory listing and metadata.

    Attributes:
        entries: List of directory entries matching the criteria.
        count: Number of entries returned.
        path: The resolved absolute path of the directory that was listed.

    Example:
        >>> result = ModelListDirectoryResult(
        ...     entries=[
        ...         ModelDirectoryEntry(
        ...             name="file.txt",
        ...             path="/data/file.txt",
        ...             is_file=True,
        ...             is_dir=False,
        ...             is_symlink=False,
        ...             size=100,
        ...             modified=1704067200.0,
        ...         ),
        ...     ],
        ...     count=1,
        ...     path="/data",
        ... )
        >>> print(result.count)
        1
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    entries: list[ModelDirectoryEntry] = Field(
        description="List of directory entries.",
    )
    count: int = Field(
        ge=0,
        description="Number of entries returned.",
    )
    path: str = Field(
        min_length=1,
        description="Resolved absolute path of the directory.",
    )


__all__: list[str] = ["ModelListDirectoryResult"]
