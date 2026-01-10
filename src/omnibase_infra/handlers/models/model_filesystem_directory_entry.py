# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Directory Entry Model.

This module provides the Pydantic model for a single directory entry,
used in the filesystem.list_directory operation result.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDirectoryEntry(BaseModel):
    """Metadata for a single directory entry.

    Represents a file or directory within a listed directory.

    Attributes:
        name: The name of the entry (filename or directory name).
        path: The full path to the entry.
        is_file: True if the entry is a regular file.
        is_dir: True if the entry is a directory.
        is_symlink: True if the entry is a symbolic link.
        size: Size in bytes (0 for directories).
        modified: Last modification time as Unix timestamp (seconds since epoch).

    Example:
        >>> entry = ModelDirectoryEntry(
        ...     name="config.json",
        ...     path="/data/config.json",
        ...     is_file=True,
        ...     is_dir=False,
        ...     is_symlink=False,
        ...     size=1024,
        ...     modified=1704067200.0,
        ... )
        >>> print(entry.is_file)
        True
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    name: str = Field(
        min_length=1,
        description="Name of the entry (filename or directory name).",
    )
    path: str = Field(
        min_length=1,
        description="Full path to the entry.",
    )
    is_file: bool = Field(
        description="True if the entry is a regular file.",
    )
    is_dir: bool = Field(
        description="True if the entry is a directory.",
    )
    is_symlink: bool = Field(
        description="True if the entry is a symbolic link.",
    )
    size: int = Field(
        ge=0,
        description="Size in bytes (0 for directories).",
    )
    modified: float = Field(
        description="Last modification time as Unix timestamp.",
    )


__all__: list[str] = ["ModelDirectoryEntry"]
