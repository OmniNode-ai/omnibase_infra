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

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

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

        Performs security validation to prevent injection attacks, directory
        traversal attacks, and ensure cross-platform compatibility.

        Path Separator Normalization:
            Path separators are normalized to forward slashes before validation.
            This ensures consistent handling of both Unix-style paths (data/file.txt)
            and Windows-style paths (data\\file.txt), as well as mixed separators
            (data/subdir\\file.txt). The original path string is preserved in the
            return value; normalization is only used for validation logic.

        Validations (in order):
            - No empty or whitespace-only paths
            - No parent directory traversal (..) - prevents sandbox escape
            - No absolute paths (/ or drive letters like C:) - enforces relative paths
            - No null bytes (prevents injection attacks)
            - No control characters (prevents terminal injection)
            - Max path length 4096 characters
            - Max filename length 255 characters
            - No reserved Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
              Trailing dots and spaces are stripped before checking, matching
              Windows filesystem behavior (e.g., "CON ", "CON.", "CON. " are all invalid)

        Args:
            v: The path string to validate.

        Returns:
            The validated path string (unchanged from input).

        Raises:
            ValueError: If any security validation fails, including:
                - Path cannot be empty or whitespace-only
                - Path contains parent directory traversal (..)
                - Absolute paths are not allowed
                - Path contains null bytes
                - Path contains control characters
                - Path exceeds maximum length
                - Filename exceeds maximum length
                - Path contains reserved Windows device name
        """
        # Check for empty or whitespace-only paths
        if not v or not v.strip():
            raise ValueError("Path cannot be empty or whitespace-only")

        # Normalize path separators for security checks (fail fast on traversal)
        normalized = v.replace("\\", "/")

        # Check for path traversal (../) - CRITICAL: prevents sandbox escape
        if ".." in normalized.split("/"):
            raise ValueError("Path contains parent directory traversal (..)")

        # Check for absolute paths - enforces relative paths only
        # Unix absolute paths start with /
        # Windows absolute paths have drive letter like C: or D:
        if v.startswith("/") or (len(v) >= 2 and v[1] == ":"):
            raise ValueError("Absolute paths are not allowed")

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
        # Normalize path separators to handle both Unix (/) and Windows (\) styles,
        # as well as mixed separators (data/subdir\file.txt)
        normalized_path = v.replace("\\", "/")
        segments = normalized_path.split("/")
        filename = segments[-1] if segments else v
        if len(filename) > 255:
            raise ValueError(
                f"Filename exceeds maximum length of 255 characters: {len(filename)}"
            )

        # Check for reserved Windows device names
        # Check both filename and filename without extension
        # Strip trailing dots and spaces before checking reserved names
        # Windows treats "CON ", "CON.", "CON. " identically to "CON"
        filename_upper = filename.upper().rstrip(". ")
        filename_base = (
            filename_upper.split(".")[0] if "." in filename_upper else filename_upper
        )
        if (
            filename_upper in _RESERVED_WINDOWS_NAMES
            or filename_base in _RESERVED_WINDOWS_NAMES
        ):
            raise ValueError(f"Path contains reserved Windows device name: {filename}")

        return v

    @field_validator("content")
    @classmethod
    def validate_content_byte_size(cls, v: str | None) -> str | None:
        """Validate content size in bytes to prevent memory exhaustion attacks.

        Pydantic's max_length validates character count, not byte size.
        Multi-byte UTF-8 characters (like emoji) can bypass the character limit
        while exceeding memory constraints. This validator ensures the actual
        byte size does not exceed 10MB.

        Security Rationale:
            A string of 10 million emoji characters passes Pydantic's max_length
            validation (10M chars <= 10M limit) but encodes to approximately
            40MB in UTF-8 (4 bytes per emoji). This enables memory exhaustion
            attacks that bypass the intended 10MB security limit.

        Args:
            v: The content string to validate, or None.

        Returns:
            The validated content string, or None.

        Raises:
            ValueError: If content exceeds 10MB in UTF-8 encoding.
        """
        if v is None:
            return v
        byte_size = len(v.encode("utf-8"))
        if byte_size > 10485760:
            raise ValueError(
                f"Content exceeds 10MB byte limit: {byte_size} bytes "
                f"(max: 10485760 bytes)"
            )
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
