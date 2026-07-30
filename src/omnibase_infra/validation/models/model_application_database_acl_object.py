# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Concrete owned object in a generated application-database ACL matrix."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.types.type_application_database_function_signature import (
    ApplicationDatabaseFunctionSignature,
)


class ModelApplicationDatabaseAclObject(BaseModel):
    """One concrete database object projected from typed ownership evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    database_ref: str = Field(..., min_length=1)
    physical_database: str = Field(..., min_length=1)
    schema_ref: str = Field(..., min_length=1)
    domain: EnumDatabaseSchemaDomain
    object_type: EnumDatabaseGrantObjectType
    object_ref: str = Field(..., min_length=1)
    catalog_kind: Literal[
        "table",
        "view",
        "materialized_view",
        "sequence",
        "function",
        "procedure",
        "type",
    ]
    owner: str = Field(..., min_length=1)
    owner_declaration: str = Field(..., min_length=1)
    target_materialized: bool
    function_signature: ApplicationDatabaseFunctionSignature | None = None
    source_keys: tuple[str, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_catalog_identity(self) -> ModelApplicationDatabaseAclObject:
        """Tie the grant target to one exact PostgreSQL catalog object shape."""
        expected_object_type = {
            "table": EnumDatabaseGrantObjectType.TABLE,
            "view": EnumDatabaseGrantObjectType.TABLE,
            "materialized_view": EnumDatabaseGrantObjectType.TABLE,
            "sequence": EnumDatabaseGrantObjectType.SEQUENCE,
            "function": EnumDatabaseGrantObjectType.FUNCTION,
            "procedure": EnumDatabaseGrantObjectType.FUNCTION,
            "type": EnumDatabaseGrantObjectType.TYPE,
        }[self.catalog_kind]
        if self.object_type is not expected_object_type:
            raise ValueError(
                f"catalog_kind={self.catalog_kind!r} requires "
                f"object_type={expected_object_type.value!r}"
            )
        is_routine = self.catalog_kind in {"function", "procedure"}
        if not is_routine and self.function_signature is not None:
            raise ValueError(
                "function_signature is only valid for function/procedure catalog kinds"
            )
        return self

    @property
    def identity(
        self,
    ) -> tuple[str, str, EnumDatabaseGrantObjectType, str, str | None]:
        """Return the exact PostgreSQL object identity, including overloads."""
        return (
            self.database_ref,
            self.schema_ref,
            self.object_type,
            self.object_ref,
            self.function_signature,
        )


__all__ = ["ModelApplicationDatabaseAclObject"]
