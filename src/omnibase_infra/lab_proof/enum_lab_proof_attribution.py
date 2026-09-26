# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Who a failing step is attributed to: the harness (inconclusive) or the subject pull request (fail).

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofAttribution(StrEnum):
    """Who a failing step is attributed to: the harness (inconclusive) or the subject pull request (fail)."""

    HARNESS = "harness"
    SUBJECT = "subject"


__all__ = ["EnumLabProofAttribution"]
