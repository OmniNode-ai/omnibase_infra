# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Ensure Directory Payload Model.

This module provides the Pydantic model for the filesystem.ensure_directory operation
payload, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEnsureDirectoryPayload(BaseModel):
    """Payload for filesystem.ensure_directory operation.

    Specifies the directory to create and options.

    Attributes:
        path: Absolute or relative path to the directory to ensure.
            The path will be validated against the allowed_paths whitelist.
            Parent directories will be created as needed (like mkdir -p).
        exist_ok: If True (default), don't raise an error if the directory
            already exists. If False, raise an error if it exists.

    Example:
        >>> payload = ModelEnsureDirectoryPayload(path="/data/output/reports")
        >>> print(payload.exist_ok)
        True

        >>> strict_payload = ModelEnsureDirectoryPayload(
        ...     path="/data/new_dir",
        ...     exist_ok=False,
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
        description="Path to the directory to ensure. Validated against allowed_paths.",
    )
    exist_ok: bool = Field(
        default=True,
        description="If True, don't error if directory already exists.",
    )


__all__: list[str] = ["ModelEnsureDirectoryPayload"]
