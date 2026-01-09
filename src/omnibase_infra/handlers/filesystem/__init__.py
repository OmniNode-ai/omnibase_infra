# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FileSystem Handler Models.

This module provides models for the FileSystem effect handler, which performs
filesystem operations including read, write, list, delete, and directory
management.

Models:
    - EnumFileSystemOperation: Operation type enum (read/write/list/delete/mkdir)
    - ModelFileSystemRequest: Request model for filesystem operations
    - ModelFileSystemResult: Result model for filesystem operations

Related:
    - OMN-1158: FileSystemHandler Implementation
    - OMN-1160: FileSystem Handler contract

Example:
    >>> from uuid import uuid4
    >>> from omnibase_infra.handlers.filesystem import (
    ...     EnumFileSystemOperation,
    ...     ModelFileSystemRequest,
    ...     ModelFileSystemResult,
    ... )
    >>> request = ModelFileSystemRequest(
    ...     operation=EnumFileSystemOperation.READ,
    ...     path="config/settings.yaml",
    ...     correlation_id=uuid4(),
    ... )
    >>> request.operation
    <EnumFileSystemOperation.READ: 'read'>
"""

from omnibase_infra.handlers.filesystem.enum_file_system_operation import (
    EnumFileSystemOperation,
)
from omnibase_infra.handlers.filesystem.model_file_system_request import (
    ModelFileSystemRequest,
)
from omnibase_infra.handlers.filesystem.model_file_system_result import (
    ModelFileSystemResult,
)

__all__: list[str] = [
    "EnumFileSystemOperation",
    "ModelFileSystemRequest",
    "ModelFileSystemResult",
]
