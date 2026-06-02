# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Aggregate run result for the DLQ replay node (OMN-12619).

This is the COMPUTE-safe aggregate the node handler returns inside
``ModelHandlerOutput[ModelDlqReplayRunResult]``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_replay_result import (
    ModelReplayResult,
)


class ModelDlqReplayRunResult(BaseModel):
    """Aggregate outcome of one replay run.

    Counts are exact, derived from per-message results, and JSON-ledger-safe so
    the EFFECT handler can return this as a typed COMPUTE-style result for
    durable evidence. ``quarantined`` proves non-replayable messages were
    routed to the quarantine topic rather than dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dlq_topic: str = Field(..., description="DLQ source topic that was drained.")
    total_processed: int = Field(..., ge=0, description="Messages consumed this run.")
    completed: int = Field(..., ge=0, description="Successfully replayed.")
    quarantined: int = Field(
        ..., ge=0, description="Non-replayable messages routed to quarantine."
    )
    failed: int = Field(..., ge=0, description="Replay attempts that failed.")
    pending: int = Field(..., ge=0, description="Dry-run would-replay count.")
    dry_run: bool = Field(..., description="Whether this run published nothing.")
    results: tuple[ModelReplayResult, ...] = Field(
        default=(), description="Per-message results in processing order."
    )


__all__ = ["ModelDlqReplayRunResult"]
