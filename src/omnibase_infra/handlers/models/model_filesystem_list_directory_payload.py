# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem List Directory Payload Model.

This module provides the Pydantic model for the filesystem.list_directory operation
payload, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelListDirectoryPayload(BaseModel):
    """Payload for filesystem.list_directory operation.

    Specifies the directory to list and filtering options.

    Attributes:
        path: Absolute or relative path to the directory to list.
            The path will be validated against the allowed_paths whitelist.
        recursive: If True, recursively list subdirectories.
            If False (default), only list immediate children.
        pattern: Optional glob pattern to filter entries by name.
            Uses fnmatch-style patterns (e.g., "*.py", "test_*").

    Example:
        >>> payload = ModelListDirectoryPayload(path="/data")
        >>> print(payload.recursive)
        False

        >>> filtered_payload = ModelListDirectoryPayload(
        ...     path="/src",
        ...     recursive=True,
        ...     pattern="*.py",
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    path: str = Field(
        min_length=1,
        description="Path to the directory to list. Validated against allowed_paths.",
    )
    recursive: bool = Field(
        default=False,
        description="If True, recursively list subdirectories.",
    )
    pattern: str | None = Field(
        default=None,
        description="Optional glob pattern to filter entries (fnmatch-style).",
    )


__all__: list[str] = ["ModelListDirectoryPayload"]
