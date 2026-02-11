# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Output model for checkpoint effect operations.

Ticket: OMN-2143
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint


class ModelCheckpointEffectOutput(BaseModel):
    """Result of a checkpoint effect node operation.

    Warning:
        **Non-standard __bool__ behavior**: Returns ``True`` only when
        ``success`` is True. Differs from typical Pydantic behavior.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the operation completed successfully.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed from the input.",
    )

    # ── write_checkpoint result ──────────────────────────────────────
    checkpoint_path: str | None = Field(
        default=None,
        description="Relative path of the written checkpoint file.",
    )

    # ── read_checkpoint result ───────────────────────────────────────
    checkpoint: ModelCheckpoint | None = Field(
        default=None,
        description="Deserialized checkpoint (read_checkpoint only).",
    )

    # ── list_checkpoints result ──────────────────────────────────────
    checkpoints: tuple[ModelCheckpoint, ...] = Field(
        default_factory=tuple,
        description="All checkpoints found (list_checkpoints only).",
    )

    # ── error information ────────────────────────────────────────────
    error: str | None = Field(
        default=None,
        description="Error message if the operation failed.",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            ``success`` is True. Differs from typical Pydantic behavior.
        """
        return self.success


__all__: list[str] = ["ModelCheckpointEffectOutput"]
