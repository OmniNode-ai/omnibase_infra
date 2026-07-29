# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Strict rich relation row emitted by the OMN-15423 inventory projection."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.enums.enum_application_inventory_object_kind import (
    EnumApplicationInventoryObjectKind,
)
from omnibase_infra.validation.models.model_internal_tenant_column_transform import (
    ModelInternalTenantColumnTransform,
)


class ModelApplicationInventoryRelation(BaseModel):
    """One repository/live-census object with its complete typed evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    kind: EnumApplicationInventoryObjectKind
    target_schema: str = Field(..., min_length=1)
    domain: EnumDatabaseSchemaDomain | None = None
    owner_declaration: str | None = None
    producer: str | None = None
    readers: tuple[str, ...] = ()
    writers: tuple[str, ...] = ()
    accessor_nodes: tuple[str, ...] = ()
    authoritative_sources: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    classification_evidence: str = Field(..., min_length=1)
    classification_status: str = Field(..., min_length=1)
    constraints: tuple[str, ...] = ()
    contract_sources: tuple[str, ...] = ()
    current_schema: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    dependent_objects: tuple[str, ...] = ()
    dsn_consumers: tuple[str, ...] = ()
    foreign_keys: tuple[str, ...] = ()
    grants: tuple[str, ...] = ()
    indexes: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    migration_root: str = Field(..., min_length=1)
    migration_stream: str | None = None
    partitioning: tuple[str, ...] = ()
    internal_tenant_column_transform: ModelInternalTenantColumnTransform | None = None


__all__ = ["ModelApplicationInventoryRelation"]
