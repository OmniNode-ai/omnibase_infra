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
    REQUIRED_GROUP_MISSING = "required_group_missing"
    REQUIRED_REFERENCE_MISSING = "required_reference_missing"
    UNEXPECTED_REFERENCE = "unexpected_reference"
    HEALTH_ASSERTION_MISSING = "health_assertion_missing"
    PROVENANCE_MISSING = "provenance_missing"
    ATTESTATION_INVALID = "attestation_invalid"
    LIVE_ATTESTATION_UNAVAILABLE = "live_attestation_unavailable"
    OBSERVATION_STALE = "observation_stale"
    REVIEWER_CLI_UNAVAILABLE = "reviewer_cli_unavailable"


__all__ = ["EnumModelReviewCapabilityFailure"]
