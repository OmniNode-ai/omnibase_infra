# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""An optional predicate a profile variant adds to its path globs.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofMatchPredicate(StrEnum):
    """An optional predicate a profile variant adds to its path globs."""

    RUNTIME_AFFECTING = "runtime_affecting"
    NOT_RUNTIME_AFFECTING = "not_runtime_affecting"


__all__ = ["EnumLabProofMatchPredicate"]
