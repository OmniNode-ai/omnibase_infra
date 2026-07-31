# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Relation evidence from a migration ownership manifest."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.enums.enum_application_relation_kind import (
    EnumApplicationRelationKind,
)


class ModelRelationEvidence(BaseModel):
    """Ownership-relevant subset of an OMN-15423 relation-evidence entry."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str = Field(..., min_length=1)
    kind: EnumApplicationRelationKind
    database_ref: str | None = None
    schema: str | None = None  # type: ignore[assignment]
    current_schemas: tuple[str, ...] = ()
    domain: EnumDatabaseSchemaDomain
    owner_declaration: str = Field(..., min_length=1)
    readers: tuple[str, ...] = ()
    writers: tuple[str, ...] = ()
    dependent_objects: tuple[str, ...] = ()

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip().lower().replace("-", "_").replace(" ", "_")


__all__ = ["ModelRelationEvidence"]
