# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed runtime evidence from the OMN-15423 inventory projection."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.validation.models.model_runtime_evidence_status import (
    ModelRuntimeEvidenceStatus,
)


class ModelApplicationInventoryRuntimeEvidence(BaseModel):
    """Runtime facts whose blocked status must survive inventory projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dsn_key_provenance: dict[str, tuple[str, ...]]
    full_day_datname_usename_activity: ModelRuntimeEvidenceStatus
    live_catalog_parity: ModelRuntimeEvidenceStatus


__all__ = ["ModelApplicationInventoryRuntimeEvidence"]
