# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FileSystem Operation Type Enum.

Defines the discriminator enum for filesystem operation types, used in the
FileSystem handler request model to specify the operation to perform.
"""

from __future__ import annotations

from enum import Enum


class EnumFileSystemOperation(str, Enum):
    """FileSystem operation type discriminator.

    Each value corresponds to a specific filesystem operation type that
    the FileSystem handler can perform.

    Attributes:
        READ: Read file contents from the filesystem.
        WRITE: Write content to a file (create or overwrite).
        LIST: List directory contents.
        DELETE: Delete a file or directory.
        MKDIR: Create a directory (optionally with parents).
    """

    READ = "read"
    WRITE = "write"
    LIST = "list"
    DELETE = "delete"
    MKDIR = "mkdir"


__all__: list[str] = ["EnumFileSystemOperation"]
