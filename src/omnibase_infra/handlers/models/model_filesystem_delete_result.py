# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Delete File Result Model.

This module provides the Pydantic model for the filesystem.delete_file operation
result, replacing untyped dict[str, object] patterns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDeleteFileResult(BaseModel):
    """Result from filesystem.delete_file operation.

    Contains metadata about the delete operation.

    Attributes:
        path: The resolved absolute path to the file (or where it would be).
        deleted: True if the file was actually deleted by this operation.
        was_missing: True if the file was already missing when the operation
            was called (only relevant when missing_ok=True).

    Example:
        >>> result = ModelDeleteFileResult(
        ...     path="/data/temp.txt",
        ...     deleted=True,
        ...     was_missing=False,
        ... )
        >>> print(result.deleted)
        True

        >>> missing_result = ModelDeleteFileResult(
        ...     path="/data/nonexistent.txt",
        ...     deleted=False,
        ...     was_missing=True,
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
        description="Resolved absolute path to the file.",
    )
    deleted: bool = Field(
        description="True if the file was deleted by this operation.",
    )
    was_missing: bool = Field(
        description="True if the file was already missing.",
    )

    @model_validator(mode="after")
    def _validate_flags(self) -> ModelDeleteFileResult:
        """Validate that deleted and was_missing cannot both be True.

        A file cannot have been deleted by this operation if it was already
        missing when the operation was called.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If both deleted and was_missing are True.
        """
        if self.deleted and self.was_missing:
            raise ValueError("deleted and was_missing cannot both be True")
        return self


__all__: list[str] = ["ModelDeleteFileResult"]
