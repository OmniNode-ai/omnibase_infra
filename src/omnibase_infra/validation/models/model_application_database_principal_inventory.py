# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed principal census for an application database ACL matrix."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.validation.enums.enum_application_database_principal_inventory_source_kind import (
    EnumApplicationDatabasePrincipalInventorySourceKind,
)
from omnibase_infra.validation.models.model_application_database_activity_evidence import (
    ModelApplicationDatabaseActivityEvidence,
)
from omnibase_infra.validation.models.model_application_database_catalog_object_evidence import (
    ModelApplicationDatabaseCatalogObjectEvidence,
)
from omnibase_infra.validation.models.model_application_database_observed_role_state import (
    ModelApplicationDatabaseObservedRoleState,
)

_SQL_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class ModelApplicationDatabasePrincipalInventory(BaseModel):
    """Exact observed non-owner principal universe for one database."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    database_ref: str = Field(..., min_length=1)
    physical_database: str = Field(..., min_length=1)
    completion_status: str = Field(..., min_length=1)
    catalog_parity_status: str = Field(..., min_length=1)
    source_kind: EnumApplicationDatabasePrincipalInventorySourceKind
    database_owner_role: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    principal_refs: tuple[str, ...] = Field(..., min_length=1)
    absent_principal_refs: tuple[str, ...] = ()
    owner_refs: tuple[str, ...] = ()
    absent_owner_refs: tuple[str, ...] = ()
    observed_role_states: tuple[ModelApplicationDatabaseObservedRoleState, ...]
    activity_principal_refs: tuple[str, ...] = ()
    observed_schema_owners: dict[str, str] = Field(default_factory=dict)
    absent_schema_refs: tuple[str, ...] = ()
    observed_objects: tuple[ModelApplicationDatabaseCatalogObjectEvidence, ...] = ()
    live_database_read: bool
    activity_evidence: ModelApplicationDatabaseActivityEvidence | None = None
    catalog_query_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    catalog_result_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    catalog_query_source_key: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    catalog_result_source_key: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    reason: str = Field(..., min_length=1)

    @field_validator(
        "principal_refs",
        "absent_principal_refs",
        "owner_refs",
        "absent_owner_refs",
        "activity_principal_refs",
        "absent_schema_refs",
    )
    @classmethod
    def validate_principal_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicates and identifiers that cannot be safely quoted."""
        if len(set(values)) != len(values):
            raise ValueError("principal_refs must be unique")
        invalid = sorted(
            value for value in values if _SQL_IDENTIFIER.fullmatch(value) is None
        )
        if invalid:
            raise ValueError(f"principal_refs contain unsafe identifiers: {invalid!r}")
        return values

    @model_validator(mode="after")
    def validate_provenance(self) -> ModelApplicationDatabasePrincipalInventory:
        """Keep synthetic and authorized-catalog evidence distinguishable."""
        overlap = set(self.principal_refs) & set(self.absent_principal_refs)
        if overlap:
            raise ValueError(
                f"principal presence and absence evidence overlap: {sorted(overlap)!r}"
            )
        owner_overlap = set(self.owner_refs) & set(self.absent_owner_refs)
        if owner_overlap:
            raise ValueError(
                f"owner presence and absence evidence overlap: {sorted(owner_overlap)!r}"
            )
        role_kind_overlap = set(self.principal_refs).union(
            self.absent_principal_refs
        ) & set(self.owner_refs).union(self.absent_owner_refs)
        if role_kind_overlap:
            raise ValueError(
                f"principal and owner evidence overlap: {sorted(role_kind_overlap)!r}"
            )
        state_roles = [state.role for state in self.observed_role_states]
        if len(set(state_roles)) != len(state_roles):
            raise ValueError("observed role states must have unique role names")
        expected_state_roles = set(self.principal_refs).union(self.owner_refs)
        if set(state_roles) != expected_state_roles:
            raise ValueError(
                "observed role states must cover the exact present principal and "
                "managed-owner census"
            )
        classified_activity_roles = (
            set(self.principal_refs)
            .union(self.owner_refs)
            .union({self.database_owner_role})
        )
        if not set(self.activity_principal_refs) <= classified_activity_roles:
            raise ValueError(
                "activity principals must be classified as present principals, "
                "managed owners, or the observed database owner"
            )
        invalid_schema_owners = sorted(
            schema
            for schema, owner in self.observed_schema_owners.items()
            if _SQL_IDENTIFIER.fullmatch(schema) is None
            or _SQL_IDENTIFIER.fullmatch(owner) is None
        )
        if invalid_schema_owners:
            raise ValueError(
                "observed schema owners contain unsafe identifiers: "
                f"{invalid_schema_owners!r}"
            )
        schema_overlap = set(self.observed_schema_owners) & set(self.absent_schema_refs)
        if schema_overlap:
            raise ValueError(
                "schema presence and absence evidence overlap: "
                f"{sorted(schema_overlap)!r}"
            )
        object_identities = [obj.identity for obj in self.observed_objects]
        if len(set(object_identities)) != len(object_identities):
            raise ValueError("observed catalog object identities must be unique")
        unknown_object_schemas = sorted(
            {obj.schema_ref for obj in self.observed_objects}
            - set(self.observed_schema_owners)
        )
        if unknown_object_schemas:
            raise ValueError(
                "observed objects require present schema-owner evidence: "
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
                "observed schema/object owners must be classified in the present "
                f"role census: {unclassified_catalog_owners!r}"
            )
        if (
            self.source_kind
            is EnumApplicationDatabasePrincipalInventorySourceKind.AUTHORIZED_CATALOG
            and not self.live_database_read
        ):
            raise ValueError("authorized_catalog requires live_database_read=true")
        if self.source_kind is (
            EnumApplicationDatabasePrincipalInventorySourceKind.AUTHORIZED_CATALOG
        ) and (
            self.activity_evidence is None
            or self.catalog_query_sha256 is None
            or self.catalog_result_sha256 is None
            or self.catalog_query_source_key is None
            or self.catalog_result_source_key is None
        ):
            raise ValueError(
                "authorized_catalog requires durable activity and catalog query/result provenance"
            )
        if (
            self.source_kind
            is EnumApplicationDatabasePrincipalInventorySourceKind.SYNTHETIC_FIXTURE
            and self.live_database_read
        ):
            raise ValueError("synthetic_fixture cannot claim a live database read")
        if self.source_kind is (
            EnumApplicationDatabasePrincipalInventorySourceKind.SYNTHETIC_FIXTURE
        ) and any(
            item is not None
            for item in (
                self.activity_evidence,
                self.catalog_query_sha256,
                self.catalog_result_sha256,
                self.catalog_query_source_key,
                self.catalog_result_source_key,
            )
        ):
            raise ValueError(
                "synthetic_fixture cannot carry authorized-catalog provenance"
            )
        return self


__all__ = ["ModelApplicationDatabasePrincipalInventory"]
