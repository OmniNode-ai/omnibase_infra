# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The change one operation makes to one declared-state item."""

from enum import StrEnum, unique


@unique
class EnumDeclaredStateOperationKind(StrEnum):
    """What an operation does to its item."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


__all__ = ["EnumDeclaredStateOperationKind"]
