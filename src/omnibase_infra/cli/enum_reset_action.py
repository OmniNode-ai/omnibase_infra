# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Enum for demo reset action classification.

Defines the possible outcomes of a single reset action: reset, preserved,
skipped, or error.

.. versionadded:: 0.9.1
"""

from __future__ import annotations

from enum import Enum

__all__: list[str] = [
    "EnumResetAction",
]


class EnumResetAction(str, Enum):
    """Classification of a reset action."""

    RESET = "reset"
    PRESERVED = "preserved"
    SKIPPED = "skipped"
    ERROR = "error"
