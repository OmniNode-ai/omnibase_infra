# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``replay_policy`` block of the DLQ replay node's contract (OMN-24177)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_target_replay_bound import (
    ModelDlqTargetReplayBound,
)


class ModelDlqReplayPolicy(BaseModel):
    """What ``contract.yaml`` declares about which records are replayed.

    ``broadcast_topics`` may be empty but must be declared; every entry is a
    non-empty topic. ``target_replay_bound`` is required: the contract owns the
    cap, the code carries no default for it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    broadcast_topics: tuple[str, ...] = Field(...)
    target_replay_bound: ModelDlqTargetReplayBound = Field(...)


__all__ = ["ModelDlqReplayPolicy"]
