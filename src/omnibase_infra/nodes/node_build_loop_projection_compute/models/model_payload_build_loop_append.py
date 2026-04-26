# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""ModelPayloadBuildLoopAppend - intent payload for build_loop_runs persistence.

The COMPUTE node emits a ModelIntent carrying this payload after parsing the
build_loop_workflow terminal event. The EFFECT node consumes the intent and
INSERTs one row into build_loop_runs.

Design choice (intentional divergence from event_ledger): build_loop_runs is
append-only with no idempotency key. Retried terminal events surface as
duplicate rows so duplication is observable rather than silently swallowed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelPayloadBuildLoopAppend(BaseModel):
    """Payload for a single build_loop_runs INSERT.

    Attributes:
        intent_type: Discriminator literal, always "build_loop.append".
        id: Primary key for the build_loop_runs row.
        run_id: Workflow run identifier extracted from the terminal event.
        workflow_name: Name of the originating workflow (e.g. "build_loop").
        event_type: Terminal event type discriminator.
        terminal_event_at: Timestamp of the terminal event itself.
        payload: Decoded terminal-event payload preserved verbatim as JSON.
        correlation_id: Distributed-trace correlation ID, when extracted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["build_loop.append"] = Field(
        default="build_loop.append",
        description="Discriminator literal for intent routing.",
    )

    id: UUID = Field(
        default_factory=uuid4,
        description="Primary key for the build_loop_runs row.",
    )

    # Pattern-validator exemption lives in validation_exemptions.yaml (OMN-9774):
    # run_id is the externally-supplied workflow run identifier (TEXT NOT NULL in
    # migration 070), not a UUID — slug-style ids are persisted verbatim.
    run_id: str = Field(
        ...,
        min_length=1,
        description="Workflow run identifier extracted from the terminal event.",
    )

    # Pattern-validator exemption lives in validation_exemptions.yaml (OMN-9774):
    # workflow_name is a workflow-yaml name label (e.g. "build_loop"), not an
    # entity reference. Used by the (workflow_name, created_at DESC) index.
    workflow_name: str = Field(
        ...,
        min_length=1,
        description="Name of the originating workflow (e.g. 'build_loop').",
    )

    event_type: str = Field(
        ...,
        min_length=1,
        description="Terminal event type discriminator.",
    )

    terminal_event_at: datetime = Field(
        ...,
        description="Timestamp of the terminal event itself.",
    )

    payload: dict[str, JsonType] = Field(
        ...,
        description="Decoded terminal-event payload preserved verbatim as JSON.",
    )

    correlation_id: UUID | None = Field(
        default=None,
        description="Distributed-trace correlation ID, when extracted.",
    )


__all__ = ["ModelPayloadBuildLoopAppend"]
