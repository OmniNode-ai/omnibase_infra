# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Non-table database object kinds represented by migration evidence."""

from enum import StrEnum, unique


@unique
class EnumApplicationDatabaseObjectKind(StrEnum):
    """Database object kinds that may be named by an ownership manifest."""

    FUNCTION = "function"
    PROCEDURE = "procedure"
    SEQUENCE = "sequence"
    EXTENSION = "extension"
    TYPE = "type"


__all__ = ["EnumApplicationDatabaseObjectKind"]
