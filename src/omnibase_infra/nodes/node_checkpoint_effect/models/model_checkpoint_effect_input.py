# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Input model for checkpoint effect operations.

Ticket: OMN-2143
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_checkpoint_phase import EnumCheckpointPhase
from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint


class ModelCheckpointEffectInput(BaseModel):
    """Input envelope for checkpoint effect node operations.

    The ``operation`` field selects the handler (write / read / list).
    Fields not relevant to the selected operation may be ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    operation: str = Field(
        ...,
        min_length=1,
        description="Operation to perform: write_checkpoint, read_checkpoint, list_checkpoints.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for tracing.",
    )

    # ── write_checkpoint fields ──────────────────────────────────────
    checkpoint: ModelCheckpoint | None = Field(
        default=None,
        description="Full checkpoint to persist (write_checkpoint only).",
    )

    # ── read_checkpoint / list_checkpoints fields ────────────────────
    ticket_id: str | None = Field(
        default=None,
        max_length=64,
        description="Ticket identifier to read/list checkpoints for.",
    )
    run_id: UUID | None = Field(
        default=None,
        description="Pipeline run ID to scope the read/list.",
    )
    phase: EnumCheckpointPhase | None = Field(
        default=None,
        description="Specific phase to read (read_checkpoint only).",
    )

    # ── optional base directory override (testing) ───────────────────
    base_dir: str | None = Field(
        default=None,
        description="Override base checkpoint directory (for testing). Relative paths preserved.",
    )


__all__: list[str] = ["ModelCheckpointEffectInput"]
