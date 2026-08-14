# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Object kinds emitted by the OMN-15423 inventory projection."""

from enum import StrEnum, unique


@unique
class EnumApplicationInventoryObjectKind(StrEnum):
    """Typed superset of ownership-checked relations and supporting objects."""

    TABLE = "table"
    VIEW = "view"
    MATERIALIZED_VIEW = "materialized_view"
    FOREIGN_TABLE = "foreign_table"
    FUNCTION = "function"
    AGGREGATE = "aggregate"
    WINDOW_FUNCTION = "window_function"
    PROCEDURE = "procedure"
    SEQUENCE = "sequence"
    EXTENSION = "extension"
    TYPE = "type"
    BASE_TYPE = "base_type"
    RANGE_TYPE = "range_type"
    MULTIRANGE_TYPE = "multirange_type"


__all__ = ["EnumApplicationInventoryObjectKind"]
