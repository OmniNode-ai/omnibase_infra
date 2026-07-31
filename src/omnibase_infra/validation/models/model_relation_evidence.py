# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Relation evidence from a migration ownership manifest."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.enums.enum_application_database_identity_root import (
    EnumApplicationDatabaseIdentityRoot,
)
from omnibase_infra.validation.enums.enum_application_database_identity_root_operation import (
    EnumApplicationDatabaseIdentityRootOperation,
)
from omnibase_infra.validation.enums.enum_application_relation_kind import (
    EnumApplicationRelationKind,
)
from omnibase_infra.validation.types.type_application_database_function_signature import (
    ApplicationDatabaseFunctionSignature,
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
    source_tenant_provenance_contract: (
        Literal["non_authoritative_provenance"] | None
    ) = None
    tenant_identity_column: str | None = Field(
        default=None,
        pattern=r"^[a-z_][a-z0-9_]*$",
    )
    identity_root_contract: EnumApplicationDatabaseIdentityRoot | None = None
    identity_root_control_role: str | None = Field(
        default=None,
        pattern=r"^[a-z_][a-z0-9_]*$",
    )
    identity_root_control_operations: tuple[
        EnumApplicationDatabaseIdentityRootOperation, ...
    ] = ()
    canonical_policy_name: str | None = Field(
        default=None,
        pattern=r"^[a-z_][a-z0-9_]*$",
    )
    deduplication_key_columns: tuple[str, ...] | None = None
    authorization_dependency_columns: tuple[str, ...] | None = None
    write_eligibility_dependency_columns: tuple[str, ...] | None = None
    function_signature: ApplicationDatabaseFunctionSignature | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    @field_validator(
        "deduplication_key_columns",
        "authorization_dependency_columns",
        "write_eligibility_dependency_columns",
    )
    @classmethod
    def validate_dependency_columns(
        cls, values: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """Keep declared semantic dependencies exact and identifier-safe."""
        if values is None:
            return None
        if len(set(values)) != len(values):
            raise ValueError("relation dependency columns must be unique")
        invalid = [
            value
            for value in values
            if re.fullmatch(r"[a-z_][a-z0-9_]*", value) is None
        ]
        if invalid:
            raise ValueError(f"invalid relation dependency columns: {invalid!r}")
        return values

    @model_validator(mode="after")
    def validate_identity_root_contract(self) -> ModelRelationEvidence:
        """Identity-root exceptions always name their policy-bound column."""
        if (
            self.identity_root_contract is not None
            and self.tenant_identity_column is None
        ):
            raise ValueError(
                "identity_root_contract requires an explicit tenant_identity_column"
            )
        if self.identity_root_contract is not None:
            if self.identity_root_control_role is None:
                raise ValueError(
                    "identity_root_contract requires an audited control role"
                )
            if not self.identity_root_control_operations:
                raise ValueError(
                    "identity_root_contract requires declared control operations"
                )
        elif (
            self.identity_root_control_role is not None
            or self.identity_root_control_operations
        ):
            raise ValueError(
                "identity-root control authority requires identity_root_contract"
            )
        if len(set(self.identity_root_control_operations)) != len(
            self.identity_root_control_operations
        ):
            raise ValueError("identity-root control operations must be unique")
        return self


__all__ = ["ModelRelationEvidence"]
