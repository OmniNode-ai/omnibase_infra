# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Non-table database object evidence from a migration ownership manifest."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.enums.enum_application_database_object_kind import (
    EnumApplicationDatabaseObjectKind,
)
from omnibase_infra.validation.types.type_application_database_function_signature import (
    ApplicationDatabaseFunctionSignature,
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
    writers: tuple[str, ...] = ()
    current_schemas: tuple[str, ...] = ()
    function_signature: ApplicationDatabaseFunctionSignature | None = None
    audit_id: str | None = Field(default=None, min_length=1)
    definition_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    @model_validator(mode="after")
    def validate_audit_pair(self) -> Self:
        """An authoritative routine audit names both its record and exact body."""
        if (self.audit_id is None) != (self.definition_sha256 is None):
            raise ValueError(
                "database object audit_id and definition_sha256 must be declared together"
            )
        if (
            self.audit_id is not None
            and self.definition_sha256 is not None
            and self.definition_sha256 not in self.audit_id
        ):
            raise ValueError("database object audit_id must bind definition_sha256")
        return self


__all__ = ["ModelDatabaseObjectEvidence"]
