# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Target mutations captured by a reverse-delta proof."""

from enum import StrEnum


class EnumReverseDeltaOperation(StrEnum):
    """Mutation kind whose inverse is durably attested."""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


__all__ = ["EnumReverseDeltaOperation"]
