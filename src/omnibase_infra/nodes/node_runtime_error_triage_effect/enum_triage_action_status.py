# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Triage action status enum for runtime error triage (OMN-5650)."""

from __future__ import annotations

from enum import StrEnum


class EnumTriageActionStatus(StrEnum):
    """Status of the triage action."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
