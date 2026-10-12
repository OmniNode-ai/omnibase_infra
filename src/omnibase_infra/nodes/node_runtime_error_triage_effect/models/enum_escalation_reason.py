# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why a runtime error incident was escalated (OMN-21029)."""

from __future__ import annotations

from enum import StrEnum


class EnumEscalationReason(StrEnum):
    """Which clause of the contract-declared escalation rule fired."""

    OCCURRENCE_THRESHOLD = "occurrence_threshold"
    NEW_FINGERPRINT_SEVERITY = "new_fingerprint_severity"


__all__ = ["EnumEscalationReason"]
