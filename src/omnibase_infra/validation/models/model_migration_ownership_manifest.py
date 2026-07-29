# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed ownership manifest for a repository migration stream."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_infra.validation.models.model_blocked_relation import ModelBlockedRelation
from omnibase_infra.validation.models.model_database_object_evidence import (
    ModelDatabaseObjectEvidence,
)
from omnibase_infra.validation.models.model_relation_evidence import (
    ModelRelationEvidence,
)
from omnibase_infra.validation.models.model_retained_live_census import (
    ModelRetainedLiveCensus,
)
from omnibase_infra.validation.models.model_runtime_evidence_status import (
    ModelRuntimeEvidenceStatus,
)


class ModelMigrationOwnershipManifest(BaseModel):
    """Typed ownership boundary for a repository migration stream."""

    model_config = ConfigDict(frozen=True, extra="allow")

    # string-version-ok: external manifest schema version from YAML.
    schema_version: str = Field(..., min_length=1)
    service: str = Field(..., min_length=1)
    owner_declaration: str | None = Field(default=None, min_length=1)
    target_database_ref: str = Field(..., min_length=1)
    db_io: ModelDbOwnershipSubcontract
    relation_evidence: tuple[ModelRelationEvidence, ...] = ()
    database_objects: tuple[ModelDatabaseObjectEvidence, ...] = ()
    blocked_relations: tuple[ModelBlockedRelation, ...] = ()
    completion_status: str | None = None
    retained_live_census: ModelRetainedLiveCensus | None = None
    runtime_evidence: dict[str, ModelRuntimeEvidenceStatus] = Field(
        default_factory=dict
    )


__all__ = ["ModelMigrationOwnershipManifest"]
