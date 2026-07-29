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
    FUNCTION = "function"
    SEQUENCE = "sequence"
    EXTENSION = "extension"


__all__ = ["EnumApplicationInventoryObjectKind"]
