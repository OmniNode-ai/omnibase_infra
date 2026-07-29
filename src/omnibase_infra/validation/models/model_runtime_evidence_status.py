# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed runtime-evidence status from an ownership manifest."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.validation.enums.enum_ownership_evidence_status import (
    EnumOwnershipEvidenceStatus,
)


class ModelRuntimeEvidenceStatus(BaseModel):
    """Status-bearing runtime evidence entry from an ownership manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EnumOwnershipEvidenceStatus
    reason: str | None = None
    credentials_captured: bool | None = None


__all__ = ["ModelRuntimeEvidenceStatus"]
