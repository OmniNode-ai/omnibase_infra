# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Payload model for the update-outcome intent.

This payload drives HandlerUpdateOutcome which performs an UPDATE on the
delta_bundles row matching (pr_ref, head_sha) to set the final outcome.

Related Tickets:
    - OMN-3142: NodeDeltaBundleEffect implementation
    - Migration 039: delta_bundles table
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadUpdateOutcome(BaseModel):
    """Payload for the delta-bundle outcome update intent.

    Carries the fields needed to complete the bundle lifecycle by setting
    outcome, merged_at, and bundle_completed_at.

    Attributes:
        intent_type: Routing key -- always "delta_bundle.update_outcome".
        correlation_id: Correlation UUID from the originating context.
        pr_ref: PR reference string to identify the bundle.
        head_sha: Git HEAD SHA to identify the bundle.
        outcome: Final PR outcome (merged, reverted, closed).
        merged_at: Timestamp when PR was merged (None if not merged).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_type: Literal["delta_bundle.update_outcome"] = Field(
        default="delta_bundle.update_outcome",
        description="Routing key for this intent.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation UUID from the originating pipeline/session.",
    )
    pr_ref: str = Field(
        ...,
        description="PR reference string to identify the bundle.",
    )
    head_sha: str = Field(
        ...,
        description="Git HEAD SHA to identify the bundle.",
    )
    outcome: Literal["merged", "reverted", "closed"] = Field(
        ...,
        description="Final PR outcome.",
    )
    merged_at: datetime | None = Field(
        default=None,
        description="Timestamp when PR was merged (None if not merged).",
    )


__all__: list[str] = ["ModelPayloadUpdateOutcome"]
