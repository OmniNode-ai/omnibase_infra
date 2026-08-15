# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed final-state evidence for one classified application relation."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from omnibase_infra.validation.enums.enum_application_database_identity_root import (
    EnumApplicationDatabaseIdentityRoot,
)
from omnibase_infra.validation.models.model_application_database_column_state import (
    ModelApplicationDatabaseColumnState,
)
from omnibase_infra.validation.models.model_application_database_function_state import (
    ModelApplicationDatabaseFunctionState,
)
from omnibase_infra.validation.models.model_application_database_identity_root_control_state import (
    ModelApplicationDatabaseIdentityRootControlState,
)
from omnibase_infra.validation.models.model_application_database_policy_state import (
    ModelApplicationDatabasePolicyState,
)
from omnibase_infra.validation.models.model_application_database_tenant_isolation_evidence import (
    ModelApplicationDatabaseTenantIsolationEvidence,
)
from omnibase_infra.validation.models.model_application_relation_declaration import (
    ModelApplicationRelationDeclaration,
)


class ModelApplicationDatabaseRelationState(BaseModel):
    """Contract classification joined to catalog-observed security state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    declaration: ModelApplicationRelationDeclaration
    columns: tuple[ModelApplicationDatabaseColumnState, ...] = ()
    primary_key_columns: tuple[str, ...] = ()
    unique_index_column_sets: tuple[tuple[str, ...], ...] = ()
    foreign_key_column_sets: tuple[tuple[str, ...], ...] = ()
    partition_key_columns: tuple[str, ...] = ()
    deduplication_key_columns: tuple[str, ...] = ()
    authorization_dependency_columns: tuple[str, ...] = ()
    write_eligibility_dependency_columns: tuple[str, ...] = ()
    rls_enabled: bool = False
    rls_forced: bool = False
    policies: tuple[ModelApplicationDatabasePolicyState, ...] = ()
    tenant_identity_column: str | None = Field(
        default=None,
        pattern=r"^[a-z_][a-z0-9_]*$",
    )
    identity_root_contract: EnumApplicationDatabaseIdentityRoot | None = None
    identity_root_control_state: (
        ModelApplicationDatabaseIdentityRootControlState | None
    ) = None
    source_tenant_provenance_contract: (
        Literal["non_authoritative_provenance"] | None
    ) = None
    canonical_policy_name: str | None = Field(
        default=None,
        pattern=r"^[a-z_][a-z0-9_]*$",
    )
    declared_restrictive_policy_names: tuple[str, ...] = ()
    restrictive_policy_proofs: Mapping[str, str] = Field(default_factory=dict)
    security_invoker: bool | None = None
    view_tenant_isolation_evidence: (
        ModelApplicationDatabaseTenantIsolationEvidence | None
    ) = None
    function_state: ModelApplicationDatabaseFunctionState | None = None

    @field_validator("restrictive_policy_proofs")
    @classmethod
    def freeze_restrictive_policy_proofs(
        cls,
        proofs: Mapping[str, str],
    ) -> Mapping[str, str]:
        """Freeze behavioral proof references against in-place mutation."""
        return MappingProxyType(dict(proofs))

    @field_serializer("restrictive_policy_proofs")
    def serialize_restrictive_policy_proofs(
        self,
        proofs: Mapping[str, str],
    ) -> dict[str, str]:
        """Restore the JSON/YAML mapping shape for immutable proof evidence."""
        return dict(proofs)

    @model_validator(mode="after")
    def validate_unique_catalog_rows(self) -> ModelApplicationDatabaseRelationState:
        """Require exact catalog rows while leaving policy verdicts to the gate."""
        column_names = [column.name for column in self.columns]
        if len(set(column_names)) != len(column_names):
            raise ValueError("relation columns must have unique names")
        if len(set(self.primary_key_columns)) != len(self.primary_key_columns):
            raise ValueError("relation primary-key columns must be unique")
        unknown_primary_columns = set(self.primary_key_columns).difference(column_names)
        if unknown_primary_columns:
            raise ValueError(
                "relation primary key names unknown columns: "
                f"{sorted(unknown_primary_columns)!r}"
            )
        column_set_fields = {
            "unique index": self.unique_index_column_sets,
            "foreign key": self.foreign_key_column_sets,
        }
        for label, column_sets in column_set_fields.items():
            if any(not column_set for column_set in column_sets):
                raise ValueError(f"{label} column sets cannot be empty")
            if len(set(column_sets)) != len(column_sets):
                raise ValueError(f"{label} column sets must be unique")
            unknown = {
                column for column_set in column_sets for column in column_set
            }.difference(column_names)
            if unknown:
                raise ValueError(f"{label} names unknown columns: {sorted(unknown)!r}")
        semantic_column_fields = {
            "partition key": self.partition_key_columns,
            "deduplication key": self.deduplication_key_columns,
            "authorization dependency": self.authorization_dependency_columns,
            "write eligibility dependency": (self.write_eligibility_dependency_columns),
        }
        for label, columns in semantic_column_fields.items():
            if len(set(columns)) != len(columns):
                raise ValueError(f"{label} columns must be unique")
            unknown = set(columns).difference(column_names)
            if unknown:
                raise ValueError(f"{label} names unknown columns: {sorted(unknown)!r}")
        policy_names = [policy.name for policy in self.policies]
        if len(set(policy_names)) != len(policy_names):
            raise ValueError("relation policies must have unique names")
        if len(set(self.declared_restrictive_policy_names)) != len(
            self.declared_restrictive_policy_names
        ):
            raise ValueError("declared restrictive policy names must be unique")
        return self


__all__ = ["ModelApplicationDatabaseRelationState"]
