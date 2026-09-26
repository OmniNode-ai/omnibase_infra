# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The ruled exemption classes, derived from the diff and the author, never from a PR body.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofExemptClass(StrEnum):
    """The ruled exemption classes, derived from the diff and the author, never from a PR body."""

    DOCS_ONLY = "docs_only"
    BOT_CHANGE_CONTROL_COMPANION = "bot_change_control_companion"


__all__ = ["EnumLabProofExemptClass"]
