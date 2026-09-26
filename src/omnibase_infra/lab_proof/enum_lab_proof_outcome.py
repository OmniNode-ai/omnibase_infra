# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The outcome of one lab proof run.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofOutcome(StrEnum):
    """The outcome of one lab proof run."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    DEV_INHERITED = "dev_inherited"


__all__ = ["EnumLabProofOutcome"]
