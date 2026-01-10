# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Write File Payload Model.

This module provides the Pydantic model for the filesystem.write_file operation
payload, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelWriteFilePayload(BaseModel):
    """Payload for filesystem.write_file operation.

    Specifies the file to write and the content to write.

    Attributes:
        path: Absolute or relative path to the file to write.
            The path will be validated against the allowed_paths whitelist.
        content: Content to write. For binary=True, this should be a
            base64-encoded string. For binary=False, this is plain text.
        binary: If True, decode content from base64 and write as binary.
            If False (default), write as text.
        create_dirs: If True, create parent directories if they don't exist.
            If False (default), fail if parent directory doesn't exist.

    Example:
        >>> payload = ModelWriteFilePayload(
        ...     path="/data/output.txt",
        ...     content="Hello, World!",
        ... )
        >>> print(payload.create_dirs)
        False

        >>> binary_payload = ModelWriteFilePayload(
        ...     path="/data/image.png",
        ...     content="iVBORw0KGgo...",  # base64 encoded
        ...     binary=True,
        ...     create_dirs=True,
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
        description="Path to the file to write. Validated against allowed_paths.",
    )
    content: str = Field(
        description="Content to write. Base64-encoded for binary mode, "
        "plain text for non-binary mode.",
    )
    binary: bool = Field(
        default=False,
        description="If True, decode content from base64 and write as binary.",
    )
    create_dirs: bool = Field(
        default=False,
        description="If True, create parent directories if they don't exist.",
    )


__all__: list[str] = ["ModelWriteFilePayload"]
