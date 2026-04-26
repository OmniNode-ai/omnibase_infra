# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""ModelBuildLoopAppendResult - outcome of a single build_loop_runs INSERT.

build_loop_runs is append-only, so there is no `duplicate` flag — every
INSERT either succeeds or raises (in contrast to ModelLedgerAppendResult
which uses ON CONFLICT DO NOTHING for kafka-position idempotency).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelBuildLoopAppendResult(BaseModel):
    """Outcome of one build_loop_runs INSERT."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the INSERT completed without error.",
    )
    id: UUID = Field(
        ...,
        description="Primary key of the inserted row.",
    )
    # Pattern-validator exemption lives in validation_exemptions.yaml (OMN-9774):
    # see ModelPayloadBuildLoopAppend for rationale (run_id is TEXT, not UUID;
    # workflow_name is a yaml-name label, not an entity reference).
    run_id: str = Field(
        ...,
        min_length=1,
        description="Workflow run identifier persisted to the row.",
    )
    workflow_name: str = Field(
        ...,
        min_length=1,
        description="Workflow name persisted to the row.",
    )


__all__ = ["ModelBuildLoopAppendResult"]
