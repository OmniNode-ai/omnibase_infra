# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Task type enum for model routing decisions."""

from __future__ import annotations

from enum import Enum


class EnumTaskType(str, Enum):
    """Task types for model routing decisions."""

    CODE_GENERATION = "code_generation"
    REFACTORING = "refactoring"
    CODE_REVIEW = "code_review"
    REASONING = "reasoning"
    DEEP_REASONING = "deep_reasoning"
    CLASSIFICATION = "classification"
    DOCUMENTATION = "documentation"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    COMPUTER_USE = "computer_use"
    TOOL_USE = "tool_use"
    MATH = "math"
    ROUTING = "routing"
    GUI_EXECUTION = "gui_execution"
    SCREEN_UNDERSTANDING = "screen_understanding"
