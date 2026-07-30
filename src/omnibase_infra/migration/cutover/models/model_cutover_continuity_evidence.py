# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-family projection or control-plane continuity evidence."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_infra.migration.cutover.models.model_control_plane_delta_evidence import (
    ModelControlPlaneDeltaEvidence,
)
from omnibase_infra.migration.cutover.models.model_projection_replay_evidence import (
    ModelProjectionReplayEvidence,
)


class ModelCutoverContinuityEvidence(BaseModel):
    """Exactly one continuity mode is consumed according to family kind."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    projection_replays: tuple[ModelProjectionReplayEvidence, ...] = ()
    control_plane_delta: ModelControlPlaneDeltaEvidence | None = None

    @field_validator("projection_replays")
    @classmethod
    def _unique_projection_offsets(
        cls,
        value: tuple[ModelProjectionReplayEvidence, ...],
    ) -> tuple[ModelProjectionReplayEvidence, ...]:
        identities = [
            (item.projection_id, item.topic, item.partition) for item in value
        ]
        if identities != sorted(
            identities,
            key=lambda identity: (str(identity[0]), identity[1], identity[2]),
        ):
            raise ValueError("projection replay evidence must be sorted")
        if len(identities) != len(set(identities)):
            raise ValueError("projection replay evidence must be unique")
        return value


__all__ = ["ModelCutoverContinuityEvidence"]
