# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Core checkpoint model with common header and discriminated phase payload.

A checkpoint records the successful completion of a pipeline phase.  Checkpoints
are append-only: re-runs produce a new checkpoint with an incremented
``attempt_number`` rather than overwriting the previous one.

Invariants:
    - Written *after* each phase completes, never during.
    - ``artifact_paths`` must contain only relative paths (no absolute machine paths).
    - ``repo_commit_map`` keys are relative to ``~/.claude/`` or repo root.

Ticket: OMN-2143
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

from omnibase_infra.enums.enum_checkpoint_phase import EnumCheckpointPhase
from omnibase_infra.models.checkpoint.model_phase_payload_create_pr import (
    ModelPhasePayloadCreatePr,
)
from omnibase_infra.models.checkpoint.model_phase_payload_implement import (
    ModelPhasePayloadImplement,
)
from omnibase_infra.models.checkpoint.model_phase_payload_local_review import (
    ModelPhasePayloadLocalReview,
)
from omnibase_infra.models.checkpoint.model_phase_payload_pr_release_ready import (
    ModelPhasePayloadPrReleaseReady,
)
from omnibase_infra.models.checkpoint.model_phase_payload_ready_for_merge import (
    ModelPhasePayloadReadyForMerge,
)

PhasePayload = Annotated[
    Union[
        Annotated[ModelPhasePayloadImplement, Tag("implement")],
        Annotated[ModelPhasePayloadLocalReview, Tag("local_review")],
        Annotated[ModelPhasePayloadCreatePr, Tag("create_pr")],
        Annotated[ModelPhasePayloadPrReleaseReady, Tag("pr_release_ready")],
        Annotated[ModelPhasePayloadReadyForMerge, Tag("ready_for_merge")],
    ],
    Discriminator("phase"),
]
"""Discriminated union of per-phase payloads, keyed on the ``phase`` literal."""

CHECKPOINT_SCHEMA_VERSION = "1.0.0"
"""Current schema version for checkpoint files."""


class ModelCheckpoint(BaseModel):
    """Immutable record of a completed pipeline phase.

    The common header fields (``schema_version`` through ``attempt_number``)
    are present on every checkpoint.  The ``phase_payload`` carries
    phase-specific data needed to resume the pipeline from this point.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # ── Common header ────────────────────────────────────────────────
    schema_version: str = Field(
        default=CHECKPOINT_SCHEMA_VERSION,
        description="Forward-compatibility version string.",
    )
    run_id: UUID = Field(
        ...,
        description="Correlation ID for the entire pipeline run.",
    )
    ticket_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Linear ticket identifier (e.g. OMN-2143).",
    )
    phase: EnumCheckpointPhase = Field(
        ...,
        description="Which pipeline phase completed.",
    )
    timestamp_utc: datetime = Field(
        ...,
        description="UTC timestamp when the phase completed.",
    )
    repo_commit_map: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of relative repo path to commit SHA.",
    )
    artifact_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Relative filesystem paths serving as resume evidence.",
    )
    attempt_number: int = Field(
        default=1,
        ge=1,
        description="Monotonically increasing attempt counter (append-only).",
    )

    # ── Phase-specific payload ───────────────────────────────────────
    phase_payload: PhasePayload = Field(
        ...,
        description="Per-phase data required to resume the pipeline.",
    )

    @model_validator(mode="after")
    def _phase_matches_payload(self) -> ModelCheckpoint:
        """Enforce that the header ``phase`` matches ``phase_payload.phase``."""
        if self.phase.value != self.phase_payload.phase:
            msg = (
                f"Header phase '{self.phase.value}' does not match "
                f"payload phase '{self.phase_payload.phase}'"
            )
            raise ValueError(msg)
        return self


__all__: list[str] = ["ModelCheckpoint", "PhasePayload", "CHECKPOINT_SCHEMA_VERSION"]
