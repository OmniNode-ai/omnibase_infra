# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed model-review runner-overlay preflight result."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.enum_model_review_capability_failure import (
    EnumModelReviewCapabilityFailure,
)


class ModelModelReviewCapabilityPreflight(BaseModel):
    """Pure, fail-closed eligibility result for one runner observation."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ready: bool
    failures: tuple[EnumModelReviewCapabilityFailure, ...] = Field(
        default_factory=tuple
    )
    missing_reference_ids: tuple[UUID, ...] = Field(default_factory=tuple)


__all__ = ["ModelModelReviewCapabilityPreflight"]
