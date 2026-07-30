# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Strict typed model for the rich OMN-15423 inventory evidence artifact."""

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.validation.models.model_application_inventory_relation import (
    ModelApplicationInventoryRelation,
)
from omnibase_infra.validation.models.model_application_inventory_relation_counts import (
    ModelApplicationInventoryRelationCounts,
)
from omnibase_infra.validation.models.model_application_inventory_runtime_evidence import (
    ModelApplicationInventoryRuntimeEvidence,
)
from omnibase_infra.validation.models.model_blocked_relation import ModelBlockedRelation
from omnibase_infra.validation.models.model_retained_live_census import (
    ModelRetainedLiveCensus,
)


class ModelApplicationRelationEvidenceInventory(BaseModel):
    """Full evidence projection produced by OMN-15423 without lossy parsing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(..., min_length=1)
    ticket: str = Field(..., min_length=1)
    database_ref: str = Field(..., min_length=1)
    physical_seed_database: str = Field(..., min_length=1)
    ownership_authority: str = Field(..., min_length=1)
    inventory_projection: str = Field(..., min_length=1)
    completion_status: str = Field(..., min_length=1)
    relation_counts: ModelApplicationInventoryRelationCounts
    relations: tuple[ModelApplicationInventoryRelation, ...]
    blocked_relations: tuple[ModelBlockedRelation, ...]
    retained_live_census: ModelRetainedLiveCensus
    runtime_evidence: ModelApplicationInventoryRuntimeEvidence

    @model_validator(mode="after")
    def validate_relation_counts(self) -> "ModelApplicationRelationEvidenceInventory":
        """Reject stale aggregate evidence instead of trusting declared counts."""
        observed = Counter(relation.kind.value for relation in self.relations)
        declared = self.relation_counts.model_dump(mode="json")
        mismatches = {
            kind: (declared_count, observed.get(kind, 0))
            for kind, declared_count in declared.items()
            if declared_count is not None and declared_count != observed.get(kind, 0)
        }
        if mismatches:
            raise ValueError(
                f"relation_counts do not match typed relation rows: {mismatches!r}"
            )
        return self


__all__ = ["ModelApplicationRelationEvidenceInventory"]
