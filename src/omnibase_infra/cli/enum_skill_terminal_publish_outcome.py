# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of publishing a skill's returned terminal result (OMN-19152)."""

from __future__ import annotations

from enum import StrEnum


class EnumSkillTerminalPublishOutcome(StrEnum):
    """What happened to a skill's returned terminal result."""

    PUBLISHED = "published"
    SKIPPED_NO_RESULT = "skipped_no_result"
    SKIPPED_NO_TERMINAL_TOPIC = "skipped_no_terminal_topic"
    FAILED = "failed"
