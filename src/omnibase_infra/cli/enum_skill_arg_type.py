# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Argument value types for the declarative ``onex skill`` mapping (OMN-13097).

Each ``onex skill`` argument declares the type its raw CLI string is coerced
to before it lands in the node-input payload. Keeping this an explicit enum
(never a free-form string) means the mapping loader fails fast on an unknown
type rather than silently passing a string through.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["EnumSkillArgType"]


class EnumSkillArgType(StrEnum):
    """Coercion target for a declarative ``onex skill`` argument."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING_LIST = "string_list"
