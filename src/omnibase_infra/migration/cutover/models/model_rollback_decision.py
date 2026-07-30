# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mechanical direct-DSN rollback verdict."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.migration.cutover.enums import EnumPostCheckpointMode


class ModelRollbackDecision(BaseModel):
    """Explain why direct rollback is permitted or refused."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    family_id: UUID
    allowed: bool
    direct_dsn_rollback: bool
    post_checkpoint_mode: EnumPostCheckpointMode
    reason: str = Field(..., min_length=1)
    reverse_delta_proof_id: UUID | None = None


__all__ = ["ModelRollbackDecision"]
