# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Ensure Directory Result Model.

This module provides the Pydantic model for the filesystem.ensure_directory operation
result, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEnsureDirectoryResult(BaseModel):
    """Result from filesystem.ensure_directory operation.

    Contains metadata about the directory creation operation.

    Attributes:
        path: The resolved absolute path to the directory.
        created: True if the directory was created by this operation.
        already_existed: True if the directory already existed before
            this operation was called.

    Example:
        >>> result = ModelEnsureDirectoryResult(
        ...     path="/data/output/reports",
        ...     created=True,
        ...     already_existed=False,
        ... )
        >>> print(result.created)
        True

        >>> existing_result = ModelEnsureDirectoryResult(
        ...     path="/data/existing",
        ...     created=False,
        ...     already_existed=True,
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
        description="Resolved absolute path to the directory.",
    )
    created: bool = Field(
        description="True if the directory was created by this operation.",
    )
    already_existed: bool = Field(
        description="True if the directory already existed.",
    )


__all__: list[str] = ["ModelEnsureDirectoryResult"]
