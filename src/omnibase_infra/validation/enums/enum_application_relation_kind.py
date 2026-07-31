# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""PostgreSQL application relation kinds covered by ownership proof."""

from enum import StrEnum, unique


@unique
class EnumApplicationRelationKind(StrEnum):
    """Physical relation kinds represented in the application inventory."""

    TABLE = "table"
    VIEW = "view"
    MATERIALIZED_VIEW = "materialized_view"
    FOREIGN_TABLE = "foreign_table"
    FUNCTION = "function"


__all__ = ["EnumApplicationRelationKind"]
