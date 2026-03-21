# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Triage action enum for runtime error triage (OMN-5650)."""

from __future__ import annotations

from enum import StrEnum


class EnumTriageAction(StrEnum):
    """Actions the triage handler can take."""

    AUTO_FIXED = "AUTO_FIXED"
    TICKET_CREATED = "TICKET_CREATED"
    DEDUPED = "DEDUPED"
    ESCALATED = "ESCALATED"
