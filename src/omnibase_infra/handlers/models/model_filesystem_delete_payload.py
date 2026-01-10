# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Delete File Payload Model.

This module provides the Pydantic model for the filesystem.delete_file operation
payload, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDeleteFilePayload(BaseModel):
    """Payload for filesystem.delete_file operation.

    Specifies the file to delete and options.

    Attributes:
        path: Absolute or relative path to the file to delete.
            The path will be validated against the allowed_paths whitelist.
            Note: This operation only deletes files, not directories.
        missing_ok: If True, don't raise an error if the file doesn't exist.
            If False (default), raise an error if the file is missing.

    Example:
        >>> payload = ModelDeleteFilePayload(path="/data/temp.txt")
        >>> print(payload.missing_ok)
        False

        >>> safe_payload = ModelDeleteFilePayload(
        ...     path="/data/maybe_exists.txt",
        ...     missing_ok=True,
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
        description="Path to the file to delete. Validated against allowed_paths.",
    )
    missing_ok: bool = Field(
        default=False,
        description="If True, don't error if file doesn't exist.",
    )


__all__: list[str] = ["ModelDeleteFilePayload"]
