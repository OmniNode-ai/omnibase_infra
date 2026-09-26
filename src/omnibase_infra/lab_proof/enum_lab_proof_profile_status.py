# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Whether a profile may be run today: live, pilot (a harness failure is inconclusive), not safe, or exempt.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofProfileStatus(StrEnum):
    """Whether a profile may be run today: live, pilot (a harness failure is inconclusive), not safe, or exempt."""

    LIVE = "live"
    PILOT = "pilot"
    NOT_SAFE = "not_safe"
    EXEMPT = "exempt"


__all__ = ["EnumLabProofProfileStatus"]
