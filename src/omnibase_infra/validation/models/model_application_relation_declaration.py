# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Normalized application-relation owner or reader declaration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.enums.enum_application_relation_kind import (
    EnumApplicationRelationKind,
)
from omnibase_infra.validation.enums.enum_application_relation_purpose import (
    EnumApplicationRelationPurpose,
)
from omnibase_infra.validation.models.model_live_application_relation import (
    RelationIdentity,
)
from omnibase_infra.validation.types.type_application_database_function_signature import (
    ApplicationDatabaseFunctionSignature,
)


class ModelApplicationRelationDeclaration(BaseModel):
    """Normalized owner or reader projected from one distributed source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    database_ref: str
    schema: str  # type: ignore[assignment]
    kind: EnumApplicationRelationKind
    purpose: EnumApplicationRelationPurpose
    domain: EnumDatabaseSchemaDomain | None
    owner_declaration: str | None
    readers: tuple[str, ...] = ()
    access: Literal["read", "write", "read_write"]
    role: str
    source_path: str
    function_signature: ApplicationDatabaseFunctionSignature | None = None

    @property
    def identity(self) -> RelationIdentity:
        return (
            self.database_ref,
            self.schema,
            self.name,
            self.kind,
            self.function_signature,
        )


__all__ = ["ModelApplicationRelationDeclaration"]
