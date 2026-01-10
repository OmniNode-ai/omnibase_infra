# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Read File Payload Model.

This module provides the Pydantic model for the filesystem.read_file operation
payload, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelReadFilePayload(BaseModel):
    """Payload for filesystem.read_file operation.

    Specifies the file to read and how to interpret its contents.

    Attributes:
        path: Absolute or relative path to the file to read.
            The path will be validated against the allowed_paths whitelist.
        binary: If True, read as binary and return base64-encoded content.
            If False (default), read as text using the specified encoding.
        encoding: Text encoding to use when binary=False.
            Defaults to "utf-8". Ignored when binary=True.

    Example:
        >>> payload = ModelReadFilePayload(path="/data/config.json")
        >>> print(payload.encoding)
        'utf-8'

        >>> binary_payload = ModelReadFilePayload(
        ...     path="/data/image.png",
        ...     binary=True,
        ... )
        >>> print(binary_payload.binary)
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
        description="Path to the file to read. Validated against allowed_paths.",
    )
    binary: bool = Field(
        default=False,
        description="If True, read as binary and return base64-encoded content.",
    )
    encoding: str = Field(
        default="utf-8",
        min_length=1,
        description="Text encoding for non-binary reads. Ignored when binary=True.",
    )


__all__: list[str] = ["ModelReadFilePayload"]
