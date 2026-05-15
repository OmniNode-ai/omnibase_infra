# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Interactive onboarding step type enum."""

from __future__ import annotations

from enum import StrEnum


class EnumInteractiveStepType(StrEnum):
    CHOICE = "choice"
    MULTI_CHOICE = "multi_choice"
    TEXT = "text"
    ACTION = "action"


__all__ = ["EnumInteractiveStepType"]
