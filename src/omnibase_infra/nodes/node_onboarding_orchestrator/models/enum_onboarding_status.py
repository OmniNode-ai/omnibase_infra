# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Onboarding result status taxonomy."""

from __future__ import annotations

from enum import StrEnum


class EnumOnboardingStatus(StrEnum):
    """Onboarding result taxonomy."""

    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"


__all__ = ["EnumOnboardingStatus"]
