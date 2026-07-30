# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed application-database catalog-result evidence."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.validation.models.model_application_database_catalog_object_evidence import (
    ModelApplicationDatabaseCatalogObjectEvidence,
)
from omnibase_infra.validation.models.model_application_database_observed_role_state import (
    ModelApplicationDatabaseObservedRoleState,
)

_SQL_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class ModelApplicationDatabaseCatalogResultEvidence(BaseModel):
    """Canonical catalog-result content bound to an immutable source blob."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    database_ref: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    physical_database: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    completion_status: str = Field(..., min_length=1)
    catalog_parity_status: str = Field(..., min_length=1)
    catalog_query_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    database_owner_role: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    principal_refs: tuple[str, ...] = Field(..., min_length=1)
    absent_principal_refs: tuple[str, ...] = ()
    owner_refs: tuple[str, ...] = ()
    absent_owner_refs: tuple[str, ...] = ()
    observed_role_states: tuple[ModelApplicationDatabaseObservedRoleState, ...]
    observed_schema_owners: dict[str, str] = Field(default_factory=dict)
    absent_schema_refs: tuple[str, ...] = ()
    observed_objects: tuple[ModelApplicationDatabaseCatalogObjectEvidence, ...] = ()

    @field_validator(
        "principal_refs",
        "absent_principal_refs",
        "owner_refs",
        "absent_owner_refs",
        "absent_schema_refs",
    )
    @classmethod
    def validate_role_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique, safely quotable role names in result content."""
        if len(set(values)) != len(values):
            raise ValueError("catalog evidence role references must be unique")
        invalid = sorted(
            value for value in values if _SQL_IDENTIFIER.fullmatch(value) is None
        )
        if invalid:
            raise ValueError(
                f"catalog evidence contains unsafe role references: {invalid!r}"
            )
        return values

    @model_validator(mode="after")
    def validate_role_census(
        self,
    ) -> ModelApplicationDatabaseCatalogResultEvidence:
        """Keep positive, negative, workload, and managed-owner evidence disjoint."""
        principal_overlap = set(self.principal_refs) & set(self.absent_principal_refs)
        if principal_overlap:
            raise ValueError(
                "catalog principal presence/absence evidence overlaps: "
                f"{sorted(principal_overlap)!r}"
            )
        owner_overlap = set(self.owner_refs) & set(self.absent_owner_refs)
        if owner_overlap:
            raise ValueError(
                "catalog owner presence/absence evidence overlaps: "
                f"{sorted(owner_overlap)!r}"
            )
        role_kind_overlap = set(self.principal_refs).union(
            self.absent_principal_refs
        ) & set(self.owner_refs).union(self.absent_owner_refs)
        if role_kind_overlap:
            raise ValueError(
                "catalog principal and managed-owner evidence overlaps: "
                f"{sorted(role_kind_overlap)!r}"
            )
        state_roles = [state.role for state in self.observed_role_states]
        expected_state_roles = set(self.principal_refs).union(self.owner_refs)
        if (
            len(set(state_roles)) != len(state_roles)
            or set(state_roles) != expected_state_roles
        ):
            raise ValueError(
                "catalog observed role states must cover the exact principal and "
                "managed-owner census"
            )
        invalid_schema_owners = sorted(
            schema
            for schema, owner in self.observed_schema_owners.items()
            if _SQL_IDENTIFIER.fullmatch(schema) is None
            or _SQL_IDENTIFIER.fullmatch(owner) is None
        )
        if invalid_schema_owners:
            raise ValueError(
                "catalog observed schema owners contain unsafe identifiers: "
                f"{invalid_schema_owners!r}"
            )
        schema_overlap = set(self.observed_schema_owners) & set(self.absent_schema_refs)
        if schema_overlap:
            raise ValueError(
                "catalog schema presence/absence evidence overlaps: "
                f"{sorted(schema_overlap)!r}"
            )
        object_identities = [obj.identity for obj in self.observed_objects]
        if len(set(object_identities)) != len(object_identities):
            raise ValueError("catalog observed object identities must be unique")
        unknown_object_schemas = sorted(
            {obj.schema_ref for obj in self.observed_objects}
            - set(self.observed_schema_owners)
        )
        if unknown_object_schemas:
            raise ValueError(
                "catalog objects require observed schema-owner evidence: "
                f"{unknown_object_schemas!r}"
            )
        classified_present_roles = (
            set(self.principal_refs)
            .union(self.owner_refs)
            .union({self.database_owner_role})
        )
        unclassified_catalog_owners = sorted(
            set(self.observed_schema_owners.values()).union(
                obj.owner for obj in self.observed_objects
            )
            - classified_present_roles
        )
        if unclassified_catalog_owners:
            raise ValueError(
                "catalog schema/object owners must be classified in the present "
                f"role census: {unclassified_catalog_owners!r}"
            )
        return self


__all__ = ["ModelApplicationDatabaseCatalogResultEvidence"]
