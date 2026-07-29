# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed live application-relation inventory."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.validation.models.model_blocked_relation import ModelBlockedRelation
from omnibase_infra.validation.models.model_live_application_relation import (
    ModelLiveApplicationRelation,
)
from omnibase_infra.validation.models.model_retained_live_census import (
    ModelRetainedLiveCensus,
)
from omnibase_infra.validation.models.model_runtime_evidence_status import (
    ModelRuntimeEvidenceStatus,
)


class ModelApplicationRelationInventory(BaseModel):
    """Live inventory consumed by the global ownership validator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # string-version-ok: fixture and evidence schema version from YAML.
    schema_version: str = Field(..., min_length=1)
    relations: tuple[ModelLiveApplicationRelation, ...]
    completion_status: str | None = None
    blocked_relations: tuple[ModelBlockedRelation, ...] = ()
    retained_live_census: ModelRetainedLiveCensus | None = None
    runtime_evidence: dict[str, ModelRuntimeEvidenceStatus] = Field(
        default_factory=dict
    )
    source_relation_count: int | None = Field(default=None, ge=0)
    excluded_database_objects: tuple[str, ...] = ()


__all__ = ["ModelApplicationRelationInventory"]
