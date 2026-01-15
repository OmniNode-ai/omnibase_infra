# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FileSystem Result Model for FileSystem Handler Operations.

This module provides ModelFileSystemResult, representing the output from the
FileSystem effect handler for filesystem operations.

Architecture:
    ModelFileSystemResult captures all information returned from
    filesystem operations:
    - Success/failure status
    - Operation that was performed
    - Target path
    - Content for read operations
    - Directory entries for list operations
    - Error message on failure

    This model is returned by the FileSystem handler after executing
    the specified operation.

Related:
    - ModelFileSystemRequest: Request model for filesystem operations
    - EnumFileSystemOperation: Operation type enum
    - OMN-1158: FileSystemHandler Implementation
    - OMN-1160: FileSystem Handler contract
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.handlers.filesystem.enum_file_system_operation import (
    EnumFileSystemOperation,
)


class ModelFileSystemResult(BaseModel):
    """Result model for filesystem handler operations.

    Contains the outcome of a filesystem operation including success status,
    operation details, and any relevant data (file content for reads,
    directory entries for lists).

    Immutability:
        This model uses frozen=True to ensure results are immutable
        once created, enabling safe concurrent access and caching.

    Attributes:
        success: Whether the operation completed successfully.
        operation: The filesystem operation that was performed.
        path: Target path the operation was performed on.
        content: File content (populated for successful READ operations).
        entries: Directory entries (populated for successful LIST operations).
        error_message: Error description (populated on failure).
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.handlers.filesystem import (
        ...     ModelFileSystemResult,
        ...     EnumFileSystemOperation,
        ... )
        >>> # Successful read result
        >>> read_result = ModelFileSystemResult(
        ...     success=True,
        ...     operation=EnumFileSystemOperation.READ,
        ...     path="config/settings.yaml",
        ...     content="key: value",
        ...     correlation_id=uuid4(),
        ... )
        >>> read_result.success
        True
        >>> # Successful list result
        >>> list_result = ModelFileSystemResult(
        ...     success=True,
        ...     operation=EnumFileSystemOperation.LIST,
        ...     path="src/",
        ...     entries=["main.py", "utils.py", "models/"],
        ...     correlation_id=uuid4(),
        ... )
        >>> len(list_result.entries)
        3
        >>> # Failed operation result
        >>> error_result = ModelFileSystemResult(
        ...     success=False,
        ...     operation=EnumFileSystemOperation.READ,
        ...     path="nonexistent.txt",
        ...     error_message="File not found: nonexistent.txt",
        ...     correlation_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the operation completed successfully",
    )
    operation: EnumFileSystemOperation = Field(
        ...,
        description="The filesystem operation that was performed",
    )
    path: str = Field(
        ...,
        description="Target path the operation was performed on",
    )
    content: str | None = Field(
        default=None,
        description="File content (populated for successful READ operations)",
    )
    entries: list[str] | None = Field(
        default=None,
        description="Directory entries (populated for successful LIST operations)",
    )

    @field_validator("entries")
    @classmethod
    def validate_entries(cls, v: list[str] | None) -> list[str] | None:
        """Validate directory entries are non-empty and non-whitespace strings.

        Catches handler implementation bugs where empty or whitespace-only
        filenames are returned. Valid filenames with embedded whitespace
        (e.g., "file name.txt") are accepted.

        Empty lists are valid and represent empty directories. Only non-empty
        lists have their individual entries validated.

        Args:
            v: The entries list to validate, or None.

        Returns:
            The validated entries list, or None.

        Raises:
            ValueError: If any entry is empty or contains only whitespace.

        Examples:
            - entries=None -> valid (non-LIST operations)
            - entries=[] -> valid (empty directory)
            - entries=["file.txt"] -> valid
            - entries=[""] -> raises ValueError
            - entries=["   "] -> raises ValueError
        """
        if v is None:
            return v
        if len(v) == 0:
            return v  # Empty directory is valid
        # Only validate entries if list is non-empty
        for entry in v:
            if not entry or not entry.strip():
                raise ValueError("Directory entries cannot be empty or whitespace-only")
        return v

    error_message: str | None = Field(
        default=None,
        description="Error description (populated on failure)",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )


__all__ = ["ModelFileSystemResult"]
