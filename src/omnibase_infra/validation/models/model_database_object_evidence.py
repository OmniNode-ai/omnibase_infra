# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Non-table database object evidence from a migration ownership manifest."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.enums.enum_application_database_object_kind import (
    EnumApplicationDatabaseObjectKind,
)


class ModelDatabaseObjectEvidence(BaseModel):
    """Ownership-relevant subset of a ``database_objects`` entry."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str = Field(..., min_length=1)
    kind: EnumApplicationDatabaseObjectKind
    database_ref: str | None = None
    schema: str | None = None  # type: ignore[assignment]
    domain: EnumDatabaseSchemaDomain
    owner_declaration: str = Field(..., min_length=1)
    readers: tuple[str, ...] = ()

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip().lower().replace("-", "_").replace(" ", "_")


__all__ = ["ModelDatabaseObjectEvidence"]
