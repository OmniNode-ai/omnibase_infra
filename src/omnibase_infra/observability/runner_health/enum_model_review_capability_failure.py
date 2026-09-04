# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model-review capability preflight failure reasons."""

from __future__ import annotations

from enum import StrEnum


class EnumModelReviewCapabilityFailure(StrEnum):
    """Reasons a runner is not eligible for model review."""

    CONFIG_ABSENT = "config_absent"
    CONFIG_INACTIVE = "config_inactive"
    REQUIRED_LABEL_MISSING = "required_label_missing"
    REQUIRED_REFERENCE_MISSING = "required_reference_missing"
    HEALTH_ASSERTION_MISSING = "health_assertion_missing"


__all__ = ["EnumModelReviewCapabilityFailure"]
