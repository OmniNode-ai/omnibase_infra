# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic discovery status enumeration for bus health diagnostics."""

from __future__ import annotations

from enum import Enum


class EnumTopicStatus(str, Enum):
    """Discovery state of a topic on the broker.

    Values:
        NOT_FOUND: Topic does not exist on the broker.
        FOUND_EMPTY: Topic exists but all partitions have high == low offsets.
        FOUND_ACTIVE: Topic exists and at least one partition has messages.
    """

    NOT_FOUND = "not_found"
    """Topic does not exist on the broker."""

    FOUND_EMPTY = "found_empty"
    """Topic exists but all partitions are empty (high == low)."""

    FOUND_ACTIVE = "found_active"
    """Topic exists and at least one partition has high > low."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumTopicStatus"]
