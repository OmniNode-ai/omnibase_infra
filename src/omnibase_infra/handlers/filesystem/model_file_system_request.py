# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FileSystem Request Model for FileSystem Handler Operations.

This module provides ModelFileSystemRequest, representing the input to the
FileSystem effect handler for filesystem operations.

Architecture:
    ModelFileSystemRequest captures all information needed to perform
    filesystem operations:
    - Operation type (read/write/list/delete/mkdir)
    - Target path for the operation
    - Optional content for write operations
    - Optional recursive flag for applicable operations

    This model is consumed by the FileSystem handler to execute
    the specified operation.

Related:
    - ModelFileSystemResult: Response model for filesystem operations
    - EnumFileSystemOperation: Operation type enum
    - OMN-1158: FileSystemHandler Implementation
    - OMN-1160: FileSystem Handler contract
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.handlers.filesystem.enum_file_system_operation import (
    EnumFileSystemOperation,
)

# Reserved Windows device names that are not allowed in paths
_RESERVED_WINDOWS_NAMES: frozenset[str] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


class ModelFileSystemRequest(BaseModel):
    """Request model for filesystem handler operations.

    Contains all information needed to perform a filesystem operation.
    The FileSystem handler uses this request to execute read, write,
    list, delete, or mkdir operations.

    Immutability:
        This model uses frozen=True to ensure requests are immutable
        once created, enabling safe concurrent access.

    Attributes:
        operation: The filesystem operation to perform.
        path: Target path for the operation (relative to workspace root).
        content: Content to write (required for WRITE operations, None otherwise).
        recursive: Whether to operate recursively (for LIST, DELETE, MKDIR).
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.handlers.filesystem import (
        ...     ModelFileSystemRequest,
        ...     EnumFileSystemOperation,
        ... )
        >>> # Read operation
        >>> read_request = ModelFileSystemRequest(
        ...     operation=EnumFileSystemOperation.READ,
        ...     path="config/settings.yaml",
        ...     correlation_id=uuid4(),
        ... )
        >>> read_request.operation
        <EnumFileSystemOperation.READ: 'read'>
        >>> # Write operation
        >>> write_request = ModelFileSystemRequest(
        ...     operation=EnumFileSystemOperation.WRITE,
        ...     path="output/results.json",
        ...     content='{"status": "success"}',
        ...     correlation_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: EnumFileSystemOperation = Field(
        ...,
        description="The filesystem operation to perform",
    )
    path: str = Field(
        ...,
        description="Target path for the operation (relative to workspace root)",
    )
    content: str | None = Field(
        default=None,
        description="Content to write (required for WRITE operations)",
        max_length=10485760,  # 10MB from contract
    )
    recursive: bool | None = Field(
        default=None,
        description="Whether to operate recursively (for LIST, DELETE, MKDIR)",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for distributed tracing",
    )

    @field_validator("path")
    @classmethod
    def validate_path_security(cls, v: str) -> str:
        """Validate path meets security requirements from contract.

        Performs security validation to prevent injection attacks and ensure
        cross-platform compatibility.

        Validations:
            - No null bytes (prevents injection attacks)
            - No control characters (prevents terminal injection)
            - Max path length 4096 characters
            - Max filename length 255 characters
            - No reserved Windows device names

        Args:
            v: The path string to validate.

        Returns:
            The validated path string.

        Raises:
            ValueError: If any security validation fails.
        """
        # Check for null bytes (injection attack prevention)
        if "\x00" in v:
            raise ValueError("Path contains null bytes")

        # Check for control characters (terminal injection prevention)
        for char in v:
            if ord(char) < 32:
                raise ValueError(f"Path contains control character: ord={ord(char)}")

        # Check max path length (4096 characters)
        if len(v) > 4096:
            raise ValueError(
                f"Path exceeds maximum length of 4096 characters: {len(v)}"
            )

        # Check max filename length (255 characters for last segment)
        segments = v.split("/")
        filename = segments[-1] if segments else v
        if len(filename) > 255:
            raise ValueError(
                f"Filename exceeds maximum length of 255 characters: {len(filename)}"
            )

        # Check for reserved Windows device names
        # Check both filename and filename without extension
        filename_upper = filename.upper()
        filename_base = (
            filename_upper.split(".")[0] if "." in filename_upper else filename_upper
        )
        if (
            filename_upper in _RESERVED_WINDOWS_NAMES
            or filename_base in _RESERVED_WINDOWS_NAMES
        ):
            raise ValueError(f"Path contains reserved Windows device name: {filename}")

        return v

    @model_validator(mode="after")
    def validate_operation_requirements(self) -> ModelFileSystemRequest:
        """Validate operation-specific requirements.

        Ensures that:
            - WRITE operations have content provided
            - READ, DELETE, LIST, MKDIR operations do not have content

        Returns:
            The validated model instance.

        Raises:
            ValueError: If operation requirements are not met.
        """
        if self.operation == EnumFileSystemOperation.WRITE:
            if self.content is None:
                raise ValueError("WRITE operation requires content")
        elif self.operation in (
            EnumFileSystemOperation.READ,
            EnumFileSystemOperation.DELETE,
            EnumFileSystemOperation.LIST,
            EnumFileSystemOperation.MKDIR,
        ):
            if self.content is not None:
                raise ValueError(
                    f"{self.operation.value.upper()} operation should not have content"
                )
        return self


__all__ = ["ModelFileSystemRequest"]
