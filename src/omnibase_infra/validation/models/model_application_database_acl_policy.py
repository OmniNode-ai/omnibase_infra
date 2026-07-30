# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Independent typed domain policy for application database principals."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.validation.enums.enum_application_database_acl_policy_source_kind import (
    EnumApplicationDatabaseAclPolicySourceKind,
)
from omnibase_infra.validation.models.model_application_database_connection_policy import (
    ModelApplicationDatabaseConnectionPolicy,
)
from omnibase_infra.validation.models.model_application_database_role_state import (
    ModelApplicationDatabaseRoleState,
)

_SQL_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class ModelApplicationDatabaseAclPolicy(BaseModel):
    """Allowed schema domains declared independently from grants being checked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    database_ref: str = Field(..., min_length=1)
    physical_database: str = Field(..., min_length=1)
    completion_status: str = Field(..., min_length=1)
    source_kind: EnumApplicationDatabaseAclPolicySourceKind
    principal_domains: dict[str, tuple[EnumDatabaseSchemaDomain, ...]] = Field(
        ..., min_length=1
    )
    database_owner_role: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    migration_principal: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    migration_owner_roles: tuple[str, ...] = Field(..., min_length=1)
    governed_role_states: tuple[ModelApplicationDatabaseRoleState, ...] = Field(
        ...,
        min_length=1,
    )
    retained_administrative_principals: tuple[str, ...] = ()
    connection_policies: tuple[ModelApplicationDatabaseConnectionPolicy, ...] = Field(
        ..., min_length=1
    )
    reason: str = Field(..., min_length=1)

    @field_validator("principal_domains")
    @classmethod
    def validate_principal_refs(
        cls,
        value: dict[str, tuple[EnumDatabaseSchemaDomain, ...]],
    ) -> dict[str, tuple[EnumDatabaseSchemaDomain, ...]]:
        """Require safe principals and unique, nonempty domain declarations."""
        invalid = sorted(
            principal
            for principal in value
            if _SQL_IDENTIFIER.fullmatch(principal) is None
        )
        if invalid:
            raise ValueError(
                f"principal_domains contain unsafe identifiers: {invalid!r}"
            )
        return value

    @field_validator("migration_owner_roles")
    @classmethod
    def validate_migration_owner_roles(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique canonical owner-role identifiers."""
        if len(set(values)) != len(values):
            raise ValueError("migration_owner_roles must be unique")
        invalid = sorted(
            value for value in values if _SQL_IDENTIFIER.fullmatch(value) is None
        )
        if invalid:
            raise ValueError(
                f"migration_owner_roles contain unsafe identifiers: {invalid!r}"
            )
        return values

    @field_validator("retained_administrative_principals")
    @classmethod
    def validate_retained_administrative_principals(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Require an explicit, unique list of untouched administrative roles."""
        if len(set(values)) != len(values):
            raise ValueError("retained administrative principals must be unique")
        invalid = sorted(
            value for value in values if _SQL_IDENTIFIER.fullmatch(value) is None
        )
        if invalid:
            raise ValueError(
                "retained administrative principals contain unsafe identifiers: "
                f"{invalid!r}"
            )
        return values

    @model_validator(mode="after")
    def validate_domains(self) -> ModelApplicationDatabaseAclPolicy:
        """Reject empty or duplicate domain declarations."""
        invalid = sorted(
            principal
            for principal, domains in self.principal_domains.items()
            if not domains or len(set(domains)) != len(domains)
        )
        if invalid:
            raise ValueError(
                f"principal_domains must be nonempty and unique for {invalid!r}"
            )
        policy_refs = [policy.database_ref for policy in self.connection_policies]
        physical_databases = [
            policy.physical_database for policy in self.connection_policies
        ]
        if len(set(policy_refs)) != len(policy_refs):
            raise ValueError("connection policy database_refs must be unique")
        if len(set(physical_databases)) != len(physical_databases):
            raise ValueError("connection policy physical databases must be unique")
        governed_roles = [state.role for state in self.governed_role_states]
        if len(set(governed_roles)) != len(governed_roles):
            raise ValueError("governed role states must have unique role names")
        return self


__all__ = ["ModelApplicationDatabaseAclPolicy"]
