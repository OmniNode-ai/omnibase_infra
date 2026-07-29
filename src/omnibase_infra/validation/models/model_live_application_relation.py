# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed live-catalog application relation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.enums.enum_application_relation_kind import (
    EnumApplicationRelationKind,
)
from omnibase_infra.validation.enums.enum_application_relation_purpose import (
    EnumApplicationRelationPurpose,
)

RelationIdentity = tuple[str, str, str, EnumApplicationRelationKind]


class ModelLiveApplicationRelation(BaseModel):
    """One relation observed by a live-catalog inventory projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    database_ref: str = Field(..., min_length=1)
    schema: str = Field(..., min_length=1)  # type: ignore[assignment]
    kind: EnumApplicationRelationKind
    purpose: EnumApplicationRelationPurpose = EnumApplicationRelationPurpose.DATA
    domain: EnumDatabaseSchemaDomain | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    @property
    def identity(self) -> RelationIdentity:
        return (self.database_ref, self.schema, self.name, self.kind)


__all__ = ["ModelLiveApplicationRelation", "RelationIdentity"]
