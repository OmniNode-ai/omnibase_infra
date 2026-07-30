# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Explicit principal-target cell in an application-database ACL matrix."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_infra.validation.types.type_application_database_function_signature import (
    ApplicationDatabaseFunctionSignature,
)


class ModelApplicationDatabaseAclRow(BaseModel):
    """One explicit principal/target cell; empty privileges mean deny."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal: str = Field(..., min_length=1)
    database_ref: str = Field(..., min_length=1)
    physical_database: str = Field(..., min_length=1)
    object_type: EnumDatabaseGrantObjectType
    schema_ref: str | None = None
    object_ref: str | None = None
    function_signature: ApplicationDatabaseFunctionSignature | None = None
    privileges: tuple[EnumDatabasePrivilege, ...] = ()

    @model_validator(mode="after")
    def validate_target(self) -> ModelApplicationDatabaseAclRow:
        """Require an exact target shape and reject duplicate privileges."""
        if len(set(self.privileges)) != len(self.privileges):
            raise ValueError("ACL row privileges must be unique")
        if self.object_type is EnumDatabaseGrantObjectType.DATABASE:
            if (
                self.schema_ref is not None
                or self.object_ref is not None
                or self.function_signature is not None
            ):
                raise ValueError("DATABASE ACL rows cannot name schema/object")
        elif self.object_type is EnumDatabaseGrantObjectType.SCHEMA:
            if (
                self.schema_ref is None
                or self.object_ref is not None
                or self.function_signature is not None
            ):
                raise ValueError("SCHEMA ACL rows require schema and no object")
        elif self.schema_ref is None or self.object_ref is None:
            raise ValueError("Object ACL rows require schema and object")
        elif (
            self.object_type is not EnumDatabaseGrantObjectType.FUNCTION
            and self.function_signature is not None
        ):
            raise ValueError("Only FUNCTION ACL rows can carry function_signature")
        return self

    @property
    def identity(
        self,
    ) -> tuple[
        str,
        str,
        EnumDatabaseGrantObjectType,
        str | None,
        str | None,
        str | None,
    ]:
        return (
            self.principal,
            self.database_ref,
            self.object_type,
            self.schema_ref,
            self.object_ref,
            self.function_signature,
        )


__all__ = ["ModelApplicationDatabaseAclRow"]
