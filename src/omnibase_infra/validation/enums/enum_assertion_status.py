# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Assertion status enum for the Demo Loop Gate (OMN-2297)."""

from __future__ import annotations

from enum import Enum


class EnumAssertionStatus(str, Enum):
    """Status of a single demo loop assertion."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
