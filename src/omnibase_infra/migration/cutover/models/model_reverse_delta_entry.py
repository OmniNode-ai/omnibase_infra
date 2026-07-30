# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One target mutation with an attested inverse artifact."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.migration.cutover.enums import EnumReverseDeltaOperation

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelReverseDeltaEntry(BaseModel):
    """Bind one target sequence to a verifiable inverse artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    entry_id: UUID
    family_id: UUID
    target_sequence: int = Field(..., ge=1)
    relation: str = Field(
        ...,
        pattern=r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$",
    )
    operation: EnumReverseDeltaOperation
    primary_key_hash: str = Field(..., pattern=_SHA256_PATTERN)
    before_image_hash: str = Field(..., pattern=_SHA256_PATTERN)
    after_image_hash: str = Field(..., pattern=_SHA256_PATTERN)
    inverse_artifact_ref: str = Field(..., min_length=1, max_length=500)


__all__ = ["ModelReverseDeltaEntry"]
